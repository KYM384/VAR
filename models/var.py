import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from dct import DCT, iDCT


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        latent_size=16, patch_size=2,
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.latent_size = latent_size
        self.L = latent_size * latent_size
        
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)

        # self.time_embedding_dim = 512
        # self.time_emb = nn.Linear(self.time_embedding_dim, self.C)

        # 3. absolute position embedding (separate tensor for each grid size)
        self.grid_sides = []
        s = 1
        while s <= latent_size:
            self.grid_sides.append(s)
            s *= 2
        self.pos_embeds = nn.ParameterDict({
            str(side): nn.Parameter(torch.empty(1, side * side, self.C))
            for side in self.grid_sides
        })
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
        SIGMA_MAX = 32
        SIGMA_MIN = 0.5
        K = 200
        sigmas = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), K-1))
        sigmas = np.concatenate(([0.0], sigmas))
        # Register sigmas/freqs as (non-persistent) buffers so `.to(device)` moves them
        # onto the GPU. The per-step blur `exp(-sigma * freqs)` then runs on-device instead
        # of on a single CPU thread (which previously stalled the GPU pipeline every step,
        # followed by a blocking host->device copy). dtypes are kept exactly as before
        # (sigmas: fp64, freqs: fp32) so the produced values are numerically unchanged.
        # persistent=False: these are constants, so they are excluded from state_dict and
        # old checkpoints still load under strict=True.
        self.register_buffer('sigmas', 0.5 * torch.from_numpy(sigmas)**2, persistent=False)

        image_size = latent_size * patch_size
        # The DCT blur curriculum and the _t_to_size schedule assume 256x256 images (freqs is
        # sliced [:size] for size up to 256). arg_util/build_vae_var default patch_size=2 ->
        # image_size=32, which would silently clip; fail loudly if patch_size=16 was not passed.
        assert image_size == 256, (
            f"VAR expects latent_size*patch_size == 256 (got {latent_size}*{patch_size}={image_size}); "
            f"pass patch_size=16 (the default is 2)."
        )
        self.register_buffer('freqs', np.pi**2 * (
            torch.arange(image_size).view(1,-1) / image_size + \
            torch.arange(image_size).view(-1,1) / image_size
        ).reshape(1,1,image_size,image_size), persistent=False)

        self.pos_tC = nn.Parameter(torch.empty(K, self.C))

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @staticmethod
    def _t_to_size(t_int: int) -> int:
        """Blur timestep -> spatial resolution, matching the training chunk schedule in
        trainer.fetch_blur_tokens (half-open intervals): t<90 ->256, [90,120) ->128,
        [120,150) ->64, [150,180) ->32, >=180 ->16. The previous ceil-based formula was
        off-by-one at the exact boundaries t in {90,120,150,180} (it gave a larger size)."""
        chunk = 0 if t_int < 90 else min(1 + (int(t_int) - 90) // 30, 4)
        return 256 // (2 ** chunk)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        inp_B3HW: torch.Tensor,
        num_steps: Optional[int] = None, shift: [Optional[float]] = None,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.pos_tC.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        schedule = torch.flip(torch.arange(len(self.sigmas)), dims=[0])
        if shift is not None:
            M = schedule.max()
            schedule = schedule / M
            schedule = shift * schedule / (1 + (shift - 1) * schedule)
            schedule = (schedule * M).long()

        if num_steps is not None:
            # indces = torch.linspace(0, len(schedule)-1, num_steps).long()
            indces = torch.linspace(len(schedule)-100, len(schedule)-50, num_steps).long()
            # indces = torch.linspace(len(schedule)-200, len(schedule)-1, num_steps).long()
            schedule = schedule[indces]

        t = schedule[0]
        size = self._t_to_size(int(t))
        print(f"start size = {size}")
        # temb = self.time_emb(self.get_time_embedding(t.reshape(1).to(inp_B3HW)).unsqueeze(1))
        temb = self.pos_tC[t].reshape(1, 1, -1)
        sigma = self.sigmas[t].reshape(1,1,1,1)
        # inp_B3HW = torch.nn.functional.interpolate(inp_B3HW, size=(size, size), mode="bilinear", align_corners=False)
        dct_B3HW = DCT(inp_B3HW)[:,:,:size,:size]
        dct_B3HW = (- sigma * self.freqs[:,:,:size,:size]).exp().to(dct_B3HW) * dct_B3HW
        # amplitude-preserving low-pass: undo the 256/size DCT-crop amplification so the VAE
        # sees in-range [-1,1] images (consistent with training).
        inp_B3HW = (iDCT(dct_B3HW) * (size / 256)).float()
        # inp_B3HW = inp_B3HW.mean((2,3),True).repeat(1,1,256,256)
        history = [ inp_B3HW.clone() ]
        # continuous encoder features (no input-side quantization), matching training
        next_token_map = self.vae_proxy[0].img_to_var_input(inp_B3HW).float()
        next_token_map = next_token_map.repeat(2,1,1)
        pos_1LC = self.pos_embeds[str(size // 16)]
        if int(t) == len(self.sigmas) - 1:
            # maximal-blur seed: mirror VAR.forward's torch.where(t==len(sigmas)-1, sos, x_BLC).
            # At t=K-1 the model is trained on a class+time-only seed (word_embed and the
            # position embeddings are dropped), so feeding the blurred image tokens here would
            # be off-distribution.
            next_token_map = sos.unsqueeze(1).expand(-1, pos_1LC.shape[1], -1) + temb
        else:
            next_token_map = self.word_embed(next_token_map) + sos.unsqueeze(1) + temb + pos_1LC

        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        for i in range(len(schedule)):
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            logits_BlV = (1+cfg) * logits_BlV[:B] - cfg * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # idx_Bl = logits_BlV.argmax(-1)
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, size//16, size//16)
            inp_B3HW_next = self.vae_proxy[0].fhat_to_img(h_BChw).float()
            history.append( inp_B3HW_next.clone() )
            if i < len(schedule) - 1:
                prev_size = size                       # resolution of the current block (s_prev)
                t_next = schedule[i+1]
                sigma_next = self.sigmas[t_next].reshape(1,1,1,1)
                size = self._t_to_size(int(t_next))
                print(size, t_next)
                dct_B3HW = DCT(inp_B3HW_next)[:,:,:size,:size]
                if dct_B3HW.shape[2] != size:
                    dct_B3HW = torch.nn.functional.pad(dct_B3HW, (0, size - dct_B3HW.shape[3], 0, size - dct_B3HW.shape[2]))
                # Blur the previous step's decoded output to the next (weaker) blur level and use
                # it directly as the next input -- no band re-addition from the previous input.
                # DCT(inp_B3HW_next) is prev_size-band-limited; inverse-transforming on the size grid
                # scales amplitude by prev_size/size, so multiply by size/prev_size to preserve it.
                dct_B3HW = (- sigma_next * self.freqs[:,:,:size,:size]).exp().to(dct_B3HW) * dct_B3HW
                inp_B3HW_next = (iDCT(dct_B3HW) * (size / prev_size)).float()

                t = t_next.clone()
                # temb = self.time_emb(self.get_time_embedding(t.reshape(1).to(inp_B3HW)).unsqueeze(1))
                temb = self.pos_tC[t].reshape(1, 1, -1)

                next_token_map = self.vae_proxy[0].img_to_var_input(inp_B3HW_next).float()
                next_token_map = next_token_map.repeat(2,1,1)
                pos_1LC = self.pos_embeds[str(size // 16)]
                if int(t) == len(self.sigmas) - 1:
                    # maximal-blur seed (see step-0 note): class+time only, matching VAR.forward.
                    next_token_map = sos.unsqueeze(1).expand(-1, pos_1LC.shape[1], -1) + temb
                else:
                    next_token_map = self.word_embed(next_token_map) + sos.unsqueeze(1) + temb + pos_1LC

            history.append( inp_B3HW_next.clone() )

        import torchvision
        history = [ torch.nn.functional.interpolate(h, size=(256, 256), mode="bilinear", align_corners=False) for h in history ]
        torchvision.utils.save_image(
            torch.cat(history), "generated2.png", normalize=True, nrow=B, value_range=(-1,1),
        )

        for b in self.blocks: b.attn.kv_caching(False)
        return inp_B3HW_next.add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    """
    def get_time_embedding(self, t):
        assert len(t.shape) == 1

        half_dim = self.time_embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0).to(t)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim).to(t) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.time_embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (t.shape[0], self.time_embedding_dim)
        return emb
    """

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, t: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, L, self.Cvae). For mixed-resolution
            training L == self.L (= L_max) and shorter grids are right-padded with dummy tokens.
        :param t: time step tensor (B,)
        :param attn_bias: optional additive attention mask (B, 1, 1, L). None => full attention over
            all L tokens (single-resolution batch, e.g. the eval scripts). When given, each sample
            attends only to its own real tokens (padding keys = -inf), which is how a minibatch with
            per-sample sizes is supported.
        :return: logits BLV, V is vocab_size
        """
        B = x_BLCv_wo_first_l.shape[0]
        L = x_BLCv_wo_first_l.shape[1]
        with torch.amp.autocast('cuda', enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, L, -1)
            # sos = self.time_emb(self.get_time_embedding(t)).unsqueeze(1) + sos
            sos = self.pos_tC[t.long()].unsqueeze(1) + sos

            if attn_bias is None:
                # single-resolution batch: every sample shares one grid -> one pos-embed
                pos_BLC = self.pos_embeds[str(int(round(L ** 0.5)))]
            else:
                # mixed-resolution batch: the grid varies per sample, so gather the right pos-embed
                # by each sample's blur chunk. Build a (num_chunks, L, C) table whose row c holds
                # chunk c's grid pos-embed zero-padded to L; dummy/padding positions get a zero
                # pos-embed and are masked out by attn_bias + the loss token_mask. Stacking ALL
                # pos_embeds also means every one gets a (possibly zero) grad, so DDP never sees
                # them as unused.
                t_long = t.long()
                chunk = torch.where(
                    t_long < 90, torch.zeros_like(t_long),
                    torch.clamp(1 + (t_long - 90) // 30, max=len(self.grid_sides) - 1),
                )
                pos_rows = []
                for side in reversed(self.grid_sides):   # [16, 8, 4, 2, 1] == chunk 0 .. last
                    p = self.pos_embeds[str(side)][0]    # (side*side, C)
                    if p.shape[0] < L:
                        p = torch.cat([p, p.new_zeros(L - p.shape[0], self.C)], dim=0)
                    pos_rows.append(p)
                pos_BLC = torch.stack(pos_rows, dim=0)[chunk]   # (B, L, C)

            x_BLC = self.word_embed(x_BLCv_wo_first_l) + sos + pos_BLC
            x_BLC = torch.where(t.view(B,1,1) == len(self.sigmas)-1, sos, x_BLC)

        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # get the compute dtype that mixed precision (autocast) will use, without the
        # previous per-forward dummy 8x8 matmul.
        main_type = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x_BLC.dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        if attn_bias is not None:
            attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if isinstance(self.word_embed, nn.Linear):
            x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
        else:
            s = 0
            for p in self.word_embed.parameters():
                if p.requires_grad:
                    s += p.view(-1)[0] * 0
            x_BLC[0, 0, 0] += s
        # dummy grad so DDP never sees unused pos_embeds. Single-resolution forwards select only
        # one entry; the mixed-resolution path already grads all of them via the stacked table, so
        # this is belt-and-suspenders.
        for p in self.pos_embeds.values():
            x_BLC[0, 0, 0] += p.view(-1)[0] * 0
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
