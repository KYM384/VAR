import torch


def DCT(image: torch.Tensor) -> torch.Tensor:
    B, C, H, W = image.shape
    image_pad = torch.cat([image, image.flip((3,))], 3)
    image_pad = torch.cat([image_pad, image_pad.flip((2,))], 2)
    image_pad = image_pad.reshape(B, C, H*2, 1, W*2, 1)
    image_pad = torch.nn.functional.pad(image_pad, (1,0,0,0,1,0,0,0))
    image_pad = image_pad.reshape(B, C, H*4, W*4)
    dcted = torch.fft.rfft2(image_pad, norm="ortho").real
    return dcted[:, :, :H, :W]


def iDCT(dcted: torch.Tensor) -> torch.Tensor:
    B, C, H, W = dcted.shape
    dcted_pad = torch.nn.functional.pad(dcted, (0,1,0,1))
    dcted_pad = torch.cat([dcted_pad, -dcted_pad[:,:,:,:-1].flip((3,))], 3)
    dcted_pad = torch.cat([dcted_pad, -dcted_pad[:,:,:-1].flip((2,))], 2)
    dcted_pad = torch.cat([dcted_pad, dcted_pad[:,:,1:-1].flip((2,))], 2)
    image = torch.fft.irfft2(dcted_pad, norm="ortho").real
    return image[:, :, 1::2, 1::2][:, :, :H, :W]