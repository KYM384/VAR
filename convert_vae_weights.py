import torch

# model = VQVAE(vocab_size=16384, ch=128, z_channels=256, share_quant_resi=1)

ckpt = torch.load("imagenet_vqvae.ckpt", weights_only=False)
state_dict2 = ckpt["state_dict"]
new_state_dict2 = {}

for k in state_dict2.keys():
    if "loss" in k:
        continue

    if ".q." in k:
        continue
    elif ".v." in k:
        continue
    elif ".k." in k:
        vk = state_dict2[k]
        vq = state_dict2[k.replace(".k.", ".q.")]
        vv = state_dict2[k.replace(".k.", ".v.")]

        v = torch.cat([vk, vq, vv], dim=0)
        new_state_dict2[k.replace(".k.", ".qkv.")] = v
    else:
        new_state_dict2[k] = state_dict2[k]

new_state_dict2["quantize.ema_vocab_hit_SV"] = torch.zeros((1024, 256))

torch.save(new_state_dict2, "vae_ch128v16384z256.pth")
