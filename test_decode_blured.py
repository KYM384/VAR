from PIL import Image
import numpy as np
import torchvision
import torch
import os

from models import VAR, VQVAE, build_vae_var
from dct import DCT, iDCT


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
torch.distributed.init_process_group("nccl", rank=0, world_size=1)

model = VQVAE(vocab_size=16384, ch=128, z_channels=256, share_quant_resi=1)
model = model.to("cuda")
model.load_state_dict(torch.load("vae_ch128v16384z256.pth"))

image = Image.open("sample.jpg")
w, h = image.size
image = image.crop((0,0,w,w)).resize((256,256), Image.LANCZOS)
x = torchvision.transforms.ToTensor()(image).unsqueeze(0).to("cuda")

SIGMA_MAX = 128
SIGMA_MIN = 0.5
T = 10
sigma_schedule = 0.5 * np.exp(2 * np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), T))
sigma_schedule = np.concatenate([np.zeros(1), sigma_schedule])
sigma_schedule = torch.from_numpy(sigma_schedule).float()
res = 256
freqs = np.pi**2 * (
    torch.arange(res).view(1,res).div(res).pow(2) + \
    torch.arange(res).view(res,1).div(res).pow(2)
).reshape(1,1,res,res)

with torch.no_grad():
    x = x.repeat(len(sigma_schedule), 1, 1, 1)
    freqs = freqs.to(x.device)
    t = sigma_schedule.to(x.device).view(-1, 1, 1, 1)
    z = DCT(x)
    z = (-freqs * t).exp() * z
    x = iDCT(z)

    x = x*2 - 1
    f_hat, usages, vq_loss = model.quantize(model.quant_conv(model.encoder(x)), ret_usages=False)
    # f_hat = model.quant_conv(model.encoder(x))
    y = model.decoder(model.post_quant_conv(f_hat))

torchvision.utils.save_image(
    torch.cat([x,y]), "a.png", nrow=len(x), normalize=True, value_range=(-1,1),
)


