from tqdm import tqdm
import torchvision
import torch
import os

from models import VQVAE, build_vae_var
from utils.data import build_dataset



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

param = 0
for p in vae.parameters():
    param += p.numel()
print(f"VQVAE #params: {param/1e6:.2f} M")

param = 0
for p in var.parameters():
    param += p.numel()
print(f"VAR #params: {param/1e6:.2f} M")

ckpt = torch.load("local_output/ar-ckpt-last.pth")["trainer"]
vae.load_state_dict(ckpt["vae_local"])
var.load_state_dict(ckpt["var_wo_ddp"])

cfg = 2.0
class_labels = tuple(range(num_classes))

B = 9
shifts = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

dataset = build_dataset("/data", final_reso=256)[-1]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4)

with torch.inference_mode(), torch.autocast("cuda", torch.float32):
    for j, (inp_B3HW, label_B) in enumerate(tqdm(dataloader)):
        if local_rank > 0:
            break

        inp_B3HW = inp_B3HW.to(device)
        label_B = label_B.to(device)

        recon_list = []
        for shift in shifts:
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, label_B=label_B, inp_B3HW=inp_B3HW, cfg=cfg, top_k=600,
                num_steps=10, shift=shift,
                top_p=0.95, g_seed=0, more_smooth=False,
            )
            recon_list.append(recon_B3HW)

        all_recon = torch.cat(recon_list, dim=0)

        torchvision.utils.save_image(
            all_recon, f"generated.png", nrow=B, normalize=True, value_range=(0,1),
        )
        break
