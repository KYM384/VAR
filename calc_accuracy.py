from tqdm import tqdm
import torchvision
import torch
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

ckpt = torch.load("local_output_initial/ar-ckpt-last.pth")["trainer"]
vae.load_state_dict(ckpt["vae_local"])
var.load_state_dict(ckpt["var_wo_ddp"])

B = 16

dataset = build_dataset("/data", final_reso=256)[-1]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4)


with torch.inference_mode(), torch.autocast("cuda", torch.float32):
    for t in np.arange(0, 200, 5)[::-1]:
        total_loss = 0
        total_acc = 0

        for j, (inp_B3HW, label_B) in enumerate(dataloader):
            if j % world_size != local_rank:
                continue

            inp_B3HW_gt = inp_B3HW.to(device)
            label_B = label_B.to(device)

            t_chunk = np.ceil((t - 70) / ((130-70)//3))
            t_chunk = np.clip(t_chunk, 0, 4)
            size = 256 // 2 ** int(t_chunk)
            """
            dct_B3HW = DCT(inp_B3HW)[:,:,:size,:size]
            dct_B3HW_blured = (- var.sigmas[t].reshape(-1,1,1,1) * var.freqs[:,:,:size,:size]).exp().to(dct_B3HW) * dct_B3HW
            inp_B3HW_blured = iDCT(dct_B3HW_blured)
            inp_B3HW = iDCT(dct_B3HW)
            """
            inp_B3HW = torch.nn.functional.interpolate(inp_B3HW_gt, (size, size), mode="bilinear")
            dct_B3HW = DCT(inp_B3HW)
            dct_B3HW_blured = (- var.sigmas[t].reshape(-1,1,1,1) * var.freqs[:,:,:size,:size]).exp().to(dct_B3HW) * dct_B3HW
            inp_B3HW_blured = iDCT(dct_B3HW_blured)
            
            gt_idx_Bl = vae.img_to_idxBl(inp_B3HW_blured)
            gt_BL = vae.img_to_idxBl(inp_B3HW)
            x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)

            t_tensor = torch.tensor(t).reshape(1).repeat(x_BLCv_wo_first_l.size(0)).to(x_BLCv_wo_first_l)
            logits_BLV = var(label_B, x_BLCv_wo_first_l, t_tensor)

            loss = torch.nn.functional.cross_entropy(logits_BLV.data.view(-1, logits_BLV.size(-1)), gt_BL.view(-1))
            acc = (logits_BLV.data.argmax(dim=-1) == gt_BL).float().mean()

            total_loss += loss.item() / len(dataloader)
            total_acc += acc.item() / len(dataloader)

        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_acc_tensor = torch.tensor(total_acc, device=device)

        torch.distributed.barrier()
        torch.distributed.reduce(total_loss_tensor, dst=0)
        torch.distributed.reduce(total_acc_tensor, dst=0)

        if local_rank == 0:
            print(f"t={t}, loss={total_loss_tensor.item():.4f}, acc={total_acc_tensor.item():.2f}%")

