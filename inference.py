"""Qualitative sampling: reconstruct a batch of real val images and save a grid PNG.

Takes the first --gen-batch images of the ImageNet val split, runs
autoregressive_infer_cfg once per --shift value, and writes all results into a
single grid (one row per shift) at --output. Only rank 0 generates.

Example (same argument style as fid_50k.sh / compute_fid.py):
    torchrun --nproc_per_node=1 inference.py --num-steps 10 --shift 1.0

DP-optimized schedule (--shift is ignored, one grid row):
    torchrun --nproc_per_node=1 inference.py --num-steps 11 \
        --dp timestep_metrics/metrics.csv --dp-metric cross_entropy
"""
import argparse

import torch
import torchvision

from dp_schedule import dp_schedule
from models import build_vae_var
from utils.data import build_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # generation
    p.add_argument('--ckpt', type=str, default='local_output_padtoken/ar-ckpt-last.pth')
    p.add_argument('--gen-batch', type=int, default=9, help='batch size = number of images per grid row')
    p.add_argument('--cfg', type=float, default=2.0)
    p.add_argument('--top-k', type=int, default=600)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--num-steps', type=int, default=None, help='subsample the FULL blur schedule to this many steps (None = all 200)')
    p.add_argument('--shift', type=float, nargs='+', default=None, help='timestep-shift warp(s) of the schedule; one grid row per value (None = no warp; ignored when --dp is given)')
    p.add_argument('--dp', type=str, default=None, metavar='METRICS_CSV',
                   help='metrics.csv from eval_timestep_metrics.py; when given, the generation schedule is '
                        'computed by dynamic programming from it (starts at t=K-1, --shift is ignored, '
                        '--num-steps sets the number of schedule nodes)')
    p.add_argument('--dp-metric', type=str, default='cross_entropy',
                   choices=['accuracy', 'cross_entropy', 'l2_distance'],
                   help='CSV column the DP cost is built from (accuracy is used as 1-accuracy)')
    p.add_argument('--seed', type=int, default=0)
    # data
    p.add_argument('--data-path', type=str, default='/data')
    p.add_argument('--workers', type=int, default=4)
    # output
    p.add_argument('--output', type=str, default='generated.png')
    return p.parse_args()


def main():
    args = parse_args()
    shifts = args.shift if args.shift is not None else [None]

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    device = 'cuda'

    vae, var = build_vae_var(
        V=16384, Cvae=8, ch=128, share_quant_resi=1,
        device=device, latent_size=16, patch_size=16,
        num_classes=1000, depth=16, shared_aln=False,
    )
    if rank == 0:
        print(f'VQVAE #params: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f} M')
        print(f'VAR #params: {sum(p.numel() for p in var.parameters()) / 1e6:.2f} M')

    ckpt = torch.load(args.ckpt, map_location='cpu')['trainer']
    vae.load_state_dict(ckpt['vae_local'])
    var.load_state_dict(ckpt['var_wo_ddp'])
    del ckpt
    # eval() is required: build_vae_var leaves VAR in training mode, and
    # torch.inference_mode()/no_grad do NOT disable DropPath (stochastic depth),
    # which would otherwise randomize the output.
    vae.eval()
    var.eval()

    # DP schedule from per-timestep metrics: overrides --shift, single grid row
    t_schedule = None
    if args.dp is not None:
        assert args.num_steps is not None, '--dp needs --num-steps (the number of schedule nodes)'
        t_schedule, dp_cost = dp_schedule(args.dp, args.dp_metric, args.num_steps, K=len(var.sigmas))
        if rank == 0:
            if args.shift is not None:
                print('[inference] --shift is ignored because --dp is given')
            print(f'[inference] DP schedule (metric={args.dp_metric}, cost={dp_cost:.4f}): {t_schedule}')
        shifts = [None]

    if rank == 0:
        # the exact steps the sampler will execute (shift/subsampling/t=0-drop applied)
        for shift in shifts:
            steps = var.build_schedule(num_steps=args.num_steps, shift=shift, full_range=True, t_schedule=t_schedule)
            print(f'[inference] executed steps (shift={shift}, {len(steps)} forwards): {steps.tolist()}')

        dataset = build_dataset(args.data_path, final_reso=256)[-1]
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.gen_batch, shuffle=False, num_workers=args.workers,
        )
        inp_B3HW, label_B = next(iter(loader))
        inp_B3HW = inp_B3HW.to(device)
        label_B = label_B.to(device)

        with torch.inference_mode(), torch.autocast('cuda', torch.float32):
            recon_list = [
                var.autoregressive_infer_cfg(
                    B=label_B.shape[0], label_B=label_B, inp_B3HW=inp_B3HW,
                    num_steps=args.num_steps, shift=shift, full_range=True,
                    t_schedule=t_schedule,
                    g_seed=args.seed, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                    more_smooth=False,
                )
                for shift in shifts
            ]

        torchvision.utils.save_image(
            torch.cat(recon_list, dim=0), args.output,
            nrow=args.gen_batch, normalize=True, value_range=(0, 1),
        )
        print(f'[inference] saved {args.output} (shifts={shifts})')

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
