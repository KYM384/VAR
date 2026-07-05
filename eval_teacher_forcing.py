"""Teacher-forcing visualization: evaluate a SINGLE Transformer forward pass.

For each validation image we build several blur strengths (DCT low-pass, exactly
as in training's fetch_blur_tokens), feed the *blurred* image to the VAR in a
single teacher-forcing forward, take the argmax of the predicted token logits,
decode it back to an image, and tile everything for visual inspection.

This is NOT the autoregressive sampler (no multi-step loop, no classifier-free
guidance). It evaluates one input -> one output of the Transformer, i.e. how well
the model deblurs a given blur level in a single shot.

Launch like inference.py (rank 0 does the work):
    torchrun --nproc_per_node=1 eval_teacher_forcing.py
"""
import torch
import torch.nn.functional as F
import torchvision

from models import VQVAE, build_vae_var
from utils.data import build_dataset
from dct import DCT, iDCT


torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)


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

ckpt = torch.load("local_output_baseline/ar-ckpt-last.pth")["trainer"]
vae.load_state_dict(ckpt["vae_local"])
var.load_state_dict(ckpt["var_wo_ddp"])

vae.eval()
var.eval()
# VAR.forward applies CFG dropout via torch.rand < cond_drop_rate and does NOT gate
# it on self.training, so it fires even in eval. Disable it here so the teacher-forced
# output is the true class-conditional prediction (not a randomly null-dropped one).
var.cond_drop_rate = 0.0

B = 8

# Blur timesteps to evaluate (0 = no blur, K-1 = fully blurred). Pick a spread that
# covers every resolution chunk; edit freely. Avoid exactly K-1 (=199): at that step
# forward() uses a class+time-only seed and ignores the image tokens.
blur_steps = [10, 70, 100, 130, 160, 190]


def t_to_size(t_int: int) -> int:
    """Blur timestep -> spatial resolution, matching trainer.fetch_blur_tokens' chunk
    schedule (chunk0: t<90 ->256, [90,120) ->128, [120,150) ->64, [150,180) ->32,
    >=180 ->16). NOTE this is the training-consistent mapping; calc_accuracy.py uses a
    stale 70/20 mapping that no longer matches training."""
    chunk = 0 if t_int < 90 else min(1 + (t_int - 90) // 30, 4)
    return 256 // (2 ** chunk)


def up256(img: torch.Tensor) -> torch.Tensor:
    """Resize any (B,3,h,w) batch to 256x256 for a uniform tiling grid."""
    return F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)


if local_rank == 0:
    dataset = build_dataset("/data", final_reso=256)[-1]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4)

    with torch.inference_mode():
        inp_B3HW_gt, label_B = next(iter(dataloader))
        inp_B3HW_gt = inp_B3HW_gt.to(device)
        label_B = label_B.to(device)
        B = inp_B3HW_gt.shape[0]

        # rows are stacked along dim 0; with nrow=B each row is one "category" of B images.
        rows = [up256(inp_B3HW_gt)]   # row 0: the original full-res images

        for t in blur_steps:
            t = int(t)
            size = t_to_size(t)
            grid = size // 16

            # --- DCT low-pass blur (training builds its teacher-forcing input the same way) ---
            # Cropping the DCT to size x size and inverse-transforming scales the amplitude by
            # 256/size (the orthonormal DCT normalization is grid-size dependent), so coarse
            # levels get amplified up to 16x and saturate [-1,1]. Re-normalize by size/256 so a
            # flat image keeps its value -- an amplitude-preserving low-pass that matches the
            # no-crop blur in test_decode_blured.py and feeds the VAE in-range images.
            scale = size / 256
            dct_B3HW = DCT(inp_B3HW_gt)[:, :, :size, :size]
            blur = (- var.sigmas[t].reshape(1, 1, 1, 1) * var.freqs[:, :, :size, :size]).exp().to(dct_B3HW)
            inp_blured = (iDCT(blur * dct_B3HW) * scale).float()   # blurred input  (B,3,size,size)
            inp_clean = (iDCT(dct_B3HW) * scale).float()           # clean low-pass = teacher-forcing target

            # --- encode the blurred image with the frozen VAE (fp32, no quantization) ---
            with torch.autocast("cuda", enabled=False):
                x_BLCv = vae.img_to_var_input(inp_blured)                # (B, L, Cvae) continuous

            # --- single teacher-forcing forward through the Transformer (no CFG) ---
            # This branch's VAR.forward takes a per-sample timestep tensor (B,), one t per
            # image; here every image in the batch uses the same blur level t.
            with torch.autocast("cuda", torch.bfloat16):
                t_tensor = torch.tensor(t).reshape(1).repeat(x_BLCv.size(0)).to(x_BLCv)  # (B,)
                logits_BLV = var(label_B, x_BLCv, t_tensor)            # (B, L, V)

            # --- decode the predicted (argmax) tokens back to an image (fp32) ---
            pred_idx_Bl = logits_BLV.float().argmax(dim=-1)             # (B, L)
            with torch.autocast("cuda", enabled=False):
                h_BChw = vae.quantize.embedding(pred_idx_Bl)            # (B, L, Cvae)
                h_BChw = h_BChw.transpose(1, 2).reshape(B, vae.Cvae, grid, grid)
                pred_img = vae.fhat_to_img(h_BChw).float()             # (B,3,size,size) in [-1,1]

            rows += [up256(inp_blured), up256(pred_img), up256(inp_clean)]
            print(f"t={t:3d}  size={size:3d}  grid={grid}x{grid}  L={x_BLCv.shape[1]}")

        grid_img = torch.cat(rows, dim=0)
        torchvision.utils.save_image(
            grid_img, "teacher_forcing.png", nrow=B, normalize=True, value_range=(-1, 1),
        )
        print("saved teacher_forcing.png")
        print("row order: [originals], then per blur step -> [blurred input] / [model output] / [clean target]")
        print(f"blur_steps = {blur_steps}")
