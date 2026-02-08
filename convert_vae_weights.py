import torch

patch_nums = list(map(int, "1_2_4_8_16".split("_")))
# model = VQVAE(vocab_size=1024, ch=128, z_channels=256, quant_conv_ks=1, quant_resi=0, share_quant_resi=1, v_patch_nums=patch_nums)

ckpt = torch.load("cifar_vae.ckpt")
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

new_state_dict2["quantize.ema_vocab_hit_SV"] = torch.zeros((len(patch_nums), 1024))

torch.save(new_state_dict2, "vae_ch128v1024z256.pth")
