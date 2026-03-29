import torch


a = torch.load("vq_ds16_c2i.pt", map_location="cpu")["model"]
b = torch.load("ar-ckpt-last.pth", map_location="cpu")["trainer"]["vae_local"]

new_dict = {}

for k1 in a.keys():
    k1_org = k1
    k1 = k1.replace("encoder.conv_blocks.", "encoder.down.")
    k1 = k1.replace("decoder.conv_blocks.", "decoder.up.")
    k1 = k1.replace("res.", "block.")
    k1 = k1.replace("conv_blockssample.", "downsample.")
    k1 = k1.replace("mid.0", "mid.block_1")
    k1 = k1.replace("mid.1", "mid.attn_1")
    k1 = k1.replace("mid.2", "mid.block_2")

    if ".q." in k1:
        k1 = k1.replace(".q.", ".qkv.")
        a[k1_org] = torch.cat([
            a[k1_org],
            a[k1_org.replace(".q.", ".k.")],
            a[k1_org.replace(".q.", ".v.")],
        ], dim=0)
    if ".k." in k1 or ".v." in k1:
        continue

    new_dict[k1] = a[k1_org]

torch.save(new_dict, "vq_ds16_c2i_llamagen.pt")