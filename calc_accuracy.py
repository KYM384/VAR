from tqdm import tqdm
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import os

from models import VQVAE, build_vae_var
from utils.data import build_dataset
from dct import DCT, iDCT



torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(world_size, local_rank)


latent_size = 16
patch_size = 16
num_classes = 1000
depth = 16
device = "cuda"

vae, var = build_vae_var(
    V=16384, Cvae=8, ch=128, share_quant_resi=1,
    device=device, latent_size=latent_size, patch_size=patch_size,
    num_classes=num_classes, depth=depth, shared_aln=False,
)

ckpt = torch.load("local_output/ar-ckpt-last.pth")["trainer"]
vae.load_state_dict(ckpt["vae_local"])
var.load_state_dict(ckpt["var_wo_ddp"])

B = 8

# Codebook for distance computation between predicted and gt embeddings.
codebook = vae.quantize.embedding.weight.detach().to(device).float()
if getattr(vae.quantize, "using_znorm", False):
    codebook = F.normalize(codebook, p=2, dim=-1)

dataset = build_dataset("/data", final_reso=256)[-1]
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, sampler=sampler, num_workers=4)


with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    for t in np.arange(0, 200, 10)[::-1]:
        total_loss_sum = 0.0
        total_dist_sum = 0.0
        total_samples = 0

        for j, (inp_B3HW, label_B) in enumerate(dataloader):
            inp_B3HW_gt = inp_B3HW.to(device)
            label_B = label_B.to(device)

            # blur timestep -> resolution chunk, matching trainer.fetch_blur_tokens
            # (chunk0: t<90 ->256, [90,120) ->128, [120,150) ->64, [150,180) ->32, >=180 ->16)
            t_chunk = 0 if t < 90 else min(4, 1 + (int(t) - 90) // 30)
            size = 256 // 2 ** int(t_chunk)
            dct_B3HW = DCT(inp_B3HW_gt)[:,:,:size,:size]
            dct_B3HW_blured = (- var.sigmas[t].reshape(-1,1,1,1) * var.freqs[:,:,:size,:size]).exp().to(dct_B3HW) * dct_B3HW
            # re-normalize the cropped iDCT by size/256 (amplitude-preserving low-pass) so the
            # frozen VAE sees in-range [-1,1] images -- consistent with training/generation.
            scale = size / 256
            inp_B3HW_blured = iDCT(dct_B3HW_blured) * scale
            inp_B3HW = iDCT(dct_B3HW) * scale
            del dct_B3HW, dct_B3HW_blured

            with torch.autocast("cuda", enabled=False):
                gt_idx_Bl = vae.img_to_idxBl(inp_B3HW_blured.float())
                gt_BL = vae.img_to_idxBl(inp_B3HW.float())
                x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)

            t_tensor = torch.tensor(t).reshape(1).repeat(x_BLCv_wo_first_l.size(0)).to(x_BLCv_wo_first_l)
            logits_BLV = var(label_B, x_BLCv_wo_first_l, t_tensor)

            logits_flat = logits_BLV.data.view(-1, logits_BLV.size(-1))
            gt_flat = gt_BL.view(-1)
            loss_sum = torch.nn.functional.cross_entropy(logits_flat, gt_flat, reduction="sum")
            pred_flat = logits_flat.argmax(dim=-1)
            dists = (codebook[gt_flat.long()] - codebook[pred_flat.long()]).norm(dim=-1)

            total_loss_sum += loss_sum.item()
            total_dist_sum += dists.sum().item()
            total_samples += gt_flat.numel()

            del logits_BLV, logits_flat, gt_flat, loss_sum, pred_flat, dists
            del inp_B3HW_gt, inp_B3HW, inp_B3HW_blured, label_B
            del gt_idx_Bl, gt_BL, x_BLCv_wo_first_l, t_tensor

        torch.cuda.empty_cache()

        total_loss_tensor = torch.tensor(total_loss_sum, device=device)
        total_dist_tensor = torch.tensor(total_dist_sum, device=device, dtype=torch.float64)
        total_samples_tensor = torch.tensor(total_samples, device=device, dtype=torch.float64)

        torch.distributed.barrier()
        torch.distributed.reduce(total_loss_tensor, dst=0)
        torch.distributed.reduce(total_dist_tensor, dst=0)
        torch.distributed.reduce(total_samples_tensor, dst=0)

        if local_rank == 0:
            n = total_samples_tensor.item()
            print(f"t={t}, loss={total_loss_tensor.item()/n:.4f}, avg_dist={total_dist_tensor.item()/n:.4f}")

