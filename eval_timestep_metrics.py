"""Per-timestep teacher-forcing metrics on the ImageNet validation set.

For each evaluated blur timestep t (same DCT low-pass construction as training's
fetch_blur_tokens: input = tokens of the blurred image, target = tokens of the
clean low-pass image at the same resolution), a single teacher-forcing forward is
run and three metrics are accumulated over ALL validation tokens:

  - accuracy       : mean( argmax(logits) == gt_token )
  - cross_entropy  : mean token-level CE (nats), label_smoothing=0
  - l2_distance    : mean || codebook_emb(pred_token) - codebook_emb(gt_token) ||_2

Every rank processes a disjoint shard of the val set for every t; the per-t sums
are all-reduced and rank 0 writes <out-dir>/metrics.csv and a config.txt.
This script only writes the CSV (no matplotlib dependency) -- render the plots
afterwards with plot_timestep_metrics.py.

Launch (single node, 8 GPUs):
    torchrun --nproc_per_node=8 eval_timestep_metrics.py

Smoke test (coarse t grid, only the first batch per rank):
    torchrun --nproc_per_node=8 eval_timestep_metrics.py --t-stride 40 --max-batches 1
"""
import argparse
import csv
import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as tdist
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms import InterpolationMode, transforms

from dct import DCT, iDCT
from models import build_vae_var
from models.var import VAR
from utils.data import ImageNetValDataset, pil_loader, normalize_01_into_pm1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=str, default='local_output_padtoken/ar-ckpt-last.pth')
    p.add_argument('--data-path', type=str, default='/data')
    p.add_argument('--out-dir', type=str, default='timestep_metrics')
    p.add_argument('--num-val', type=int, default=-1, help='number of val images (-1 = all 50000)')
    p.add_argument('--batch', type=int, default=32, help='per-GPU batch size')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--t-stride', type=int, default=5, help='evaluate t = 0, s, 2s, ... (t=K-1 is always appended)')
    p.add_argument('--t-list', type=str, default=None, help='comma-separated explicit timesteps, overrides --t-stride')
    p.add_argument('--fp32', action='store_true', help='run the Transformer in fp32 instead of bf16 autocast')
    p.add_argument('--max-batches', type=int, default=-1, help='stop after this many batches per rank (-1 = whole shard); smoke tests use 1')
    return p.parse_args()


def build_val_dataset(data_path):
    # identical to utils.data.build_dataset's val_aug (deterministic, [-1, 1] range)
    mid_reso = round(1.125 * 256)
    val_transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ])
    return ImageNetValDataset(root=osp.join(data_path, 'val'), loader=pil_loader, transform=val_transform)


def main():
    args = parse_args()

    tdist.init_process_group(backend='nccl', init_method='env://')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()
    torch.cuda.set_device(rank)
    device = 'cuda'

    # ---- model ----
    vae, var = build_vae_var(
        V=16384, Cvae=8, ch=128, share_quant_resi=1,
        device=device, latent_size=16, patch_size=16,
        num_classes=1000, depth=16, shared_aln=False,
    )
    ckpt = torch.load(args.ckpt, map_location='cpu')['trainer']
    vae.load_state_dict(ckpt['vae_local'])
    var.load_state_dict(ckpt['var_wo_ddp'])
    del ckpt
    vae.eval()
    var.eval()
    # forward() applies CFG label dropout via torch.rand regardless of .training;
    # disable it so every sample is evaluated with its true class label.
    var.cond_drop_rate = 0.0

    K = len(var.sigmas)  # 200
    if args.t_list is not None:
        ts = sorted({int(s) for s in args.t_list.split(',')})
        assert all(0 <= t < K for t in ts), f'timesteps must be in [0, {K})'
    else:
        ts = list(range(0, K, args.t_stride))
        if ts[-1] != K - 1:
            ts.append(K - 1)

    # ---- data: shard a deterministic subset of val across ranks ----
    dataset = build_val_dataset(args.data_path)
    num_val = len(dataset) if args.num_val < 0 else min(args.num_val, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_val).astype(np.int64)
    my_indices = indices[rank::world_size].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, my_indices),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    codebook = vae.quantize.embedding  # (V, Cvae) token embeddings for the L2 metric

    with torch.inference_mode():
        # per-t running sums, all-reduced at the end:
        # columns = [num_tokens, num_correct, ce_sum, l2_sum]
        accum = torch.zeros(len(ts), 4, dtype=torch.float64, device=device)

        for batch_idx, (inp_B3HW, label_B) in enumerate(tqdm(loader, desc='val batches', disable=rank != 0)):
            inp_B3HW = inp_B3HW.to(device, non_blocking=True)
            label_B = label_B.to(device, non_blocking=True)
            B = inp_B3HW.shape[0]
            dct_full = DCT(inp_B3HW)  # one 256x256 DCT per batch, cropped per timestep below

            for ti, t in enumerate(ts):
                size = VAR._t_to_size(t)
                scale = size / 256  # amplitude-preserving low-pass, as in fetch_blur_tokens
                dct = dct_full[:, :, :size, :size]
                blur = (- var.sigmas[t] * var.freqs[:, :, :size, :size]).exp().to(dct)
                blured = (iDCT(blur * dct) * scale).float()  # model input
                clean = (iDCT(dct) * scale).float()          # target

                # frozen-VAE tokenization in fp32 (outside the bf16 autocast), one fused encode
                both = vae.img_to_idxBl(torch.cat([blured, clean], dim=0))
                inp_idx_Bl, gt_idx_Bl = both[:B], both[B:]          # (B, Lg) each
                x_BLCv = vae.quantize.idxBl_to_var_input(inp_idx_Bl)  # (B, Lg, Cvae)

                t_tensor = torch.full((B,), float(t), device=device)
                # single-resolution batch -> attn_bias=None, exactly like trainer.eval_ep's forward
                with torch.autocast('cuda', torch.bfloat16, enabled=not args.fp32):
                    logits_BLV = var(label_B, x_BLCv, t_tensor)     # (B, Lg, V), fp32 logits

                V = logits_BLV.shape[-1]
                logits_flat = logits_BLV.float().reshape(-1, V)
                gt_flat = gt_idx_Bl.reshape(-1).long()
                pred_flat = logits_flat.argmax(dim=-1)

                ce_sum = F.cross_entropy(logits_flat, gt_flat, reduction='sum')
                correct = (pred_flat == gt_flat).sum()
                l2_sum = (codebook(pred_flat) - codebook(gt_flat)).norm(dim=-1).sum()

                accum[ti, 0] += gt_flat.numel()
                accum[ti, 1] += correct
                accum[ti, 2] += ce_sum.double()
                accum[ti, 3] += l2_sum.double()

            # safe under distribution: the only collective is the single all_reduce
            # below, which every rank reaches exactly once however early it breaks
            if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
                break

        tdist.all_reduce(accum, op=tdist.ReduceOp.SUM)

    # ---- rank 0: CSV ----
    if rank == 0:
        accum_np = accum.cpu().numpy()
        tokens = np.maximum(accum_np[:, 0], 1.0)
        acc = accum_np[:, 1] / tokens
        ce = accum_np[:, 2] / tokens
        l2 = accum_np[:, 3] / tokens

        os.makedirs(args.out_dir, exist_ok=True)

        csv_path = osp.join(args.out_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'size', 'num_tokens', 'accuracy', 'cross_entropy', 'l2_distance'])
            for i, t in enumerate(ts):
                w.writerow([t, VAR._t_to_size(t), int(accum_np[i, 0]),
                            f'{acc[i]:.6f}', f'{ce[i]:.6f}', f'{l2[i]:.6f}'])
        print(f'[eval] wrote {csv_path}')
        print(f'[eval] render the plots with: python plot_timestep_metrics.py {csv_path}')

        with open(osp.join(args.out_dir, 'config.txt'), 'w') as f:
            f.write(f'num_val_images={num_val}\nworld_size={world_size}\ntimesteps={ts}\n')
            for k, v in sorted(vars(args).items()):
                f.write(f'{k}={v}\n')

    tdist.barrier()
    tdist.destroy_process_group()


if __name__ == '__main__':
    main()
