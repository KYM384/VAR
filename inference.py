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
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
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

ckpt = torch.load("local_output2/ar-ckpt-best.pth")["trainer"]
vae.load_state_dict(ckpt["vae_local"])
var.load_state_dict(ckpt["var_wo_ddp"])

cfg = 2.0
class_labels = tuple(range(num_classes))

B = 9

dataset = build_dataset("/data", final_reso=256)[-1]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4)

with torch.inference_mode(), torch.autocast("cuda", torch.float32):
    for j, (inp_B3HW, label_B) in enumerate(tqdm(dataloader)):
        if j % world_size != local_rank:
            continue

        inp_B3HW = inp_B3HW.to(device)
        label_B = label_B.to(device)

        recon_B3HW = var.autoregressive_infer_cfg(
            B=B, label_B=label_B, inp_B3HW=inp_B3HW, cfg=cfg, top_k=600,
            num_steps=10, shift=1.0,
            top_p=0.95, g_seed=None, more_smooth=False,
        )

        torchvision.utils.save_image(
            recon_B3HW, f"generated.png", nrow=3, normalize=True, value_range=(0,1),
        )
        break

"""
for step in [10]:
    os.makedirs(f"generated", exist_ok=True)

    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        for i in range(5):
            for j, (inp_B3HW, label_B) in enumerate(tqdm(dataloader)):
                if j % world_size != local_rank:
                    continue

                inp_B3HW = inp_B3HW.to(device)
                label_B = label_B.to(device)

                recon_B3HW = var.autoregressive_infer_cfg(
                    B=B, label_B=label_B, inp_B3HW=inp_B3HW, cfg=cfg, top_k=600,
                    num_steps=step, shift=1.0,
                    top_p=0.95, g_seed=None, more_smooth=False,
                )

                for k in range(B):
                    torchvision.utils.save_image(
                        recon_B3HW[k:k+1], f"generated/{i*len(dataset) + j*B + k:06}.png", normalize=True, value_range=(0,1),
                    )
"""
