import torchvision
import torch

from models import VQVAE, build_vae_var


patch_nums = (1,2,3,4,5,7,10,13,16)
num_classes = 10
depth = 16
device = "cuda"

vae, var = build_vae_var(
    V=1024, Cvae=256, ch=128, share_quant_resi=1,       # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
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

ckpt = torch.load("ar-ckpt-best.pth")["trainer"]
vae.load_state_dict(ckpt["vae_local"])
var.load_state_dict(ckpt["var_wo_ddp"])

seed = 0
cfg = 1.0
class_labels = tuple(range(10))
more_smooth = False

B = len(class_labels)
label_B = torch.tensor(class_labels, device=device)
with torch.inference_mode():
    with torch.autocast("cuda", torch.float16):
        recon_B3HW = var.autoregressive_infer_cfg(
            B=B, label_B=label_B, cfg=cfg, top_k=900,
            top_p=0.95, g_seed=seed, more_smooth=more_smooth,
        )

    torchvision.utils.save_image(
        recon_B3HW, "generated.png", normalize=True, value_range=(0,1),
    )
