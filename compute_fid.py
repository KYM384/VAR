"""Distributed FID evaluation WITHOUT saving generated images.

Generated images never touch the disk: each batch is sampled, pushed through the
FID InceptionV3, and only the 2048-d feature statistics (running sum + running
sum of outer products, fp64) are kept. Statistics are all-reduced across ranks
and rank 0 computes the Frechet distance.

Reference statistics are computed from real ImageNet images (same on-the-fly
scheme) and cached to an .npz under fid_stats/, so the real pass runs only once
per (split, count) combination.

Standard 50k evaluation (8 GPUs):
    torchrun --nproc_per_node=8 compute_fid.py --num-gen 50000

Smoke test (50 generated images -- statistically meaningless FID, only checks
that the pipeline runs end to end):
    torchrun --nproc_per_node=1 compute_fid.py --num-gen 50 --ref-split val --num-ref 1000

DP-optimized schedule from per-timestep metrics (eval_timestep_metrics.py CSV;
starts at t=K-1, --shift is ignored, --num-steps = number of schedule nodes):
    torchrun --nproc_per_node=8 compute_fid.py --num-gen 50000 --num-steps 11 \
        --dp timestep_metrics/metrics.csv --dp-metric cross_entropy

Prerequisite (login node, once):
    wget https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
"""
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as tdist
from tqdm import tqdm
from torchvision.transforms import InterpolationMode, transforms

from dp_schedule import dp_schedule
from fid_inception import InceptionV3
from models import build_vae_var
from utils.data import ImageNetTrainDataset, ImageNetValDataset, pil_loader

FEATURE_DIM = 2048


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # generation
    p.add_argument('--ckpt', type=str, default='local_output_padtoken/ar-ckpt-last.pth')
    p.add_argument('--num-gen', type=int, default=50000, help='number of generated samples (label i%%1000 -> exactly 50/class at 50000)')
    p.add_argument('--gen-batch', type=int, default=25, help='generation batch size per GPU')
    p.add_argument('--cfg', type=float, default=2.0)
    p.add_argument('--top-k', type=int, default=600)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--num-steps', type=int, default=None, help='subsample the FULL blur schedule to this many steps (None = all 200)')
    p.add_argument('--shift', type=float, default=None, help='timestep-shift warp of the schedule (None = no warp; ignored when --dp is given)')
    p.add_argument('--dp', type=str, default=None, metavar='METRICS_CSV',
                   help='metrics.csv from eval_timestep_metrics.py; when given, the generation schedule is '
                        'computed by dynamic programming from it (starts at t=K-1, --shift is ignored, '
                        '--num-steps sets the number of schedule nodes)')
    p.add_argument('--dp-metric', type=str, default='cross_entropy',
                   choices=['accuracy', 'cross_entropy', 'l2_distance'],
                   help='CSV column the DP cost is built from (accuracy is used as 1-accuracy)')
    p.add_argument('--seed', type=int, default=0)
    # reference statistics
    p.add_argument('--data-path', type=str, default='/data')
    p.add_argument('--ref-split', type=str, default='train', choices=['train', 'val'])
    p.add_argument('--num-ref', type=int, default=-1, help='number of real reference images (-1 = whole split)')
    p.add_argument('--ref-batch', type=int, default=64)
    p.add_argument('--ref-workers', type=int, default=8)
    p.add_argument('--ref-cache', type=str, default=None, help='override the fid_stats/... cache path')
    # inception
    p.add_argument('--inception-weights', type=str, default='pt_inception-2015-12-05-6726825d.pth')
    return p.parse_args()


class RunningStats:
    """Streaming mean/covariance in fp64; equals np.cov(feats, rowvar=False) with ddof=1."""

    def __init__(self, dim: int, device):
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, feats: torch.Tensor):
        f = feats.double()
        self.n += f.shape[0]
        self.sum += f.sum(dim=0)
        self.outer += f.t() @ f

    def all_reduce(self):
        for t in (self.n, self.sum, self.outer):
            tdist.all_reduce(t, op=tdist.ReduceOp.SUM)

    def finalize(self):
        n = self.n.item()
        assert n > 1, f'need >1 samples for a covariance, got {n}'
        mu = self.sum / n
        sigma = (self.outer - n * torch.outer(mu, mu)) / (n - 1)
        return mu.cpu().numpy(), sigma.cpu().numpy(), int(n)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """pytorch-fid's implementation (scipy sqrtm + eps fallback)."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    try:
        from scipy import linalg
    except ImportError:
        linalg = None

    if linalg is not None:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print(f'[fid] covariance product is singular; retrying with {eps} on the diagonal')
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f'imaginary component {np.max(np.abs(covmean.imag))} in sqrtm')
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
    else:
        # scipy-free fallback: tr sqrt(S1 S2) via the eigenvalues of S1 S2 (similar to a
        # PSD matrix, so its eigenvalues are real >= 0 up to numerical noise).
        print('[fid] scipy not available; using the eigenvalue fallback for tr(sqrtm)')
        eigvals = np.linalg.eigvals(sigma1.dot(sigma2))
        tr_covmean = np.sqrt(np.clip(eigvals.real, 0.0, None)).sum()

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def build_ref_dataset(args):
    """Real-image dataset with the val preprocessing (deterministic center crop) but
    WITHOUT the [-1,1] normalization: InceptionV3 here expects [0,1] input."""
    mid_reso = round(1.125 * 256)
    ref_transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),  # [0, 1]
    ])
    if args.ref_split == 'train':
        return ImageNetTrainDataset(root=osp.join(args.data_path, 'train'), loader=pil_loader, transform=ref_transform)
    return ImageNetValDataset(root=osp.join(args.data_path, 'val'), loader=pil_loader, transform=ref_transform)


def compute_ref_stats(args, inception, device, rank, world_size):
    dataset = build_ref_dataset(args)
    num_ref = len(dataset) if args.num_ref < 0 else min(args.num_ref, len(dataset))
    # evenly spaced subset (deterministic), then shard by rank
    indices = np.linspace(0, len(dataset) - 1, num_ref).astype(np.int64)
    my_indices = indices[rank::world_size].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, my_indices),
        batch_size=args.ref_batch, shuffle=False,
        num_workers=args.ref_workers, pin_memory=True, drop_last=False,
    )

    stats = RunningStats(FEATURE_DIM, device)
    for img_B3HW, _ in tqdm(loader, desc=f'ref features ({args.ref_split})', disable=rank != 0):
        stats.update(inception(img_B3HW.to(device, non_blocking=True)))
    stats.all_reduce()
    return stats.finalize()


def compute_gen_stats(args, var, inception, device, rank, world_size, t_schedule=None):
    labels_all = torch.arange(args.num_gen) % var.num_classes  # 50k -> exactly 50 per class
    labels_rank = labels_all[rank::world_size]

    stats = RunningStats(FEATURE_DIM, device)
    starts = range(0, len(labels_rank), args.gen_batch)
    for start in tqdm(starts, desc='generating', disable=rank != 0):
        label_B = labels_rank[start:start + args.gen_batch].to(device)
        # distinct, reproducible seed per (rank, batch) -- a shared seed would make
        # every rank generate the same noise sequence.
        g_seed = args.seed * 1_000_003 + rank * 100_000 + start
        img_B3HW = var.autoregressive_infer_cfg(
            B=label_B.shape[0], label_B=label_B, inp_B3HW=None,
            num_steps=None if t_schedule is not None else args.num_steps,
            shift=None if t_schedule is not None else args.shift,
            t_schedule=t_schedule, full_range=True,
            g_seed=g_seed, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
            more_smooth=False, verbose=False, save_history=None,
        )  # (B, 3, 256, 256) in [0, 1] -- never written to disk
        stats.update(inception(img_B3HW.clamp_(0, 1)))
    stats.all_reduce()
    return stats.finalize()


def main():
    args = parse_args()

    tdist.init_process_group(backend='nccl', init_method='env://')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()
    torch.cuda.set_device(rank)
    device = 'cuda'

    inception = InceptionV3(args.inception_weights).to(device)

    # ---- reference statistics (cached) ----
    if args.ref_cache is not None:
        cache_path = args.ref_cache
    else:
        tag = 'all' if args.num_ref < 0 else str(args.num_ref)
        cache_path = osp.join('fid_stats', f'imagenet256_{args.ref_split}_{tag}.npz')

    if osp.exists(cache_path):
        if rank == 0:
            data = np.load(cache_path)
            mu_ref, sigma_ref, n_ref = data['mu'], data['sigma'], int(data['num'])
            print(f'[fid] loaded reference stats from {cache_path} ({n_ref} images)')
        else:
            mu_ref = sigma_ref = n_ref = None
    else:
        mu_ref, sigma_ref, n_ref = compute_ref_stats(args, inception, device, rank, world_size)
        if rank == 0:
            os.makedirs(osp.dirname(cache_path) or '.', exist_ok=True)
            np.savez(cache_path, mu=mu_ref, sigma=sigma_ref, num=n_ref)
            print(f'[fid] saved reference stats to {cache_path} ({n_ref} images)')

    # ---- build model & load checkpoint ----
    vae, var = build_vae_var(
        V=16384, Cvae=8, ch=128, share_quant_resi=1,
        device=device, latent_size=16, patch_size=16,
        num_classes=1000, depth=16, shared_aln=False,
    )
    ckpt = torch.load(args.ckpt, map_location='cpu')['trainer']
    vae.load_state_dict(ckpt['vae_local'])
    var.load_state_dict(ckpt['var_wo_ddp'])
    del ckpt
    # eval() is required: build_vae_var leaves VAR in training mode and no_grad does not
    # disable DropPath (see inference.py).
    vae.eval()
    var.eval()

    # ---- DP schedule from per-timestep metrics (optional) ----
    t_schedule = None
    if args.dp is not None:
        assert args.num_steps is not None, '--dp needs --num-steps (the number of schedule nodes)'
        t_schedule, dp_cost = dp_schedule(args.dp, args.dp_metric, args.num_steps, K=len(var.sigmas))
        if rank == 0:
            if args.shift is not None:
                print('[fid] --shift is ignored because --dp is given')
            print(f'[fid] DP schedule (metric={args.dp_metric}, cost={dp_cost:.4f}): {t_schedule}')

    # the exact steps the sampler will execute (shift/subsampling/t=0-drop applied);
    # also covers non-DP runs, where the schedule is the uniform/shifted subsample
    exec_schedule = var.build_schedule(
        num_steps=None if t_schedule is not None else args.num_steps,
        shift=None if t_schedule is not None else args.shift,
        full_range=True, t_schedule=t_schedule,
    )
    if rank == 0:
        print(f'[fid] executed schedule ({len(exec_schedule)} forwards): {exec_schedule.tolist()}')

    # ---- generate & accumulate features (no images are saved) ----
    with torch.inference_mode():
        mu_gen, sigma_gen, n_gen = compute_gen_stats(args, var, inception, device, rank, world_size, t_schedule)

    # ---- FID on rank 0 ----
    if rank == 0:
        fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        print(
            f'\n[fid] FID = {fid:.4f}\n'
            f'      gen: {n_gen} samples | ref: {n_ref} {args.ref_split} images\n'
            f'      ckpt={args.ckpt}\n'
            f'      cfg={args.cfg} top_k={args.top_k} top_p={args.top_p} '
            f'num_steps={args.num_steps} shift={args.shift} seed={args.seed}'
        )
        if t_schedule is not None:
            print(f'      schedule=DP({args.dp_metric} from {args.dp}): {t_schedule}')
        print(f'      executed schedule ({len(exec_schedule)} forwards): {exec_schedule.tolist()}')
        if n_gen < 50000:
            print(f'      WARNING: {n_gen} < 50000 samples -- smoke-test number, not a reportable FID.')

    tdist.barrier()
    tdist.destroy_process_group()


if __name__ == '__main__':
    main()
