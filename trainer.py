import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

import numpy as np
from dct import DCT, iDCT

import wandb

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, latent_size: int,
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        fused_vae_encode: bool = False, vae_bf16: bool = False,
    ):
        super(VARTrainer, self).__init__()

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        # del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = latent_size * latent_size
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        # opt-in tokenization speed knobs (see utils/arg_util.py); default off to keep tokens identical
        self.fused_vae_encode = fused_vae_encode
        self.vae_bf16 = vae_bf16
        # number of blur-decreasing steps in the autoregressive sequence (fixed for now)
        self.N = 10

    @staticmethod
    def _t_to_size(t_int: int) -> int:
        """Map a blur timestep to the spatial resolution, matching the previous chunking:
        t in [0,90)->256, [90,120)->128, [120,150)->64, [150,180)->32, [180,...)->16."""
        chunk = 0 if t_int < 90 else min(1 + (t_int - 90) // 30, 4)
        return 256 // (2 ** chunk)

    @torch.no_grad()
    def fetch_blur_tokens(self, inp_B3HW: FTen):
        """Build an N-step *autoregressive* blur sequence and tokenize each step with the frozen VAE.

        Instead of a single blurred image, we draw one monotonically-decreasing blur schedule
        (shared across the batch) and produce N images whose blur weakens from step 0 -> N-1
        (block 0 is always t = K-1 = fully blurred / coarsest seed; blur weakens toward the finest block):

          1. v = cumsum(softmax(randn(N)))           # monotone increasing in (0, 1]
          2. t_seq = round(v * (K-1)).flip(0)         # monotone *decreasing* timesteps (most -> least blur)
                                                      # v[-1]==1 then flipped -> block 0 t == K-1 (fully blurred seed)
          3. for each t_i: resolution size_i (= _t_to_size), blur the low-pass DCT, tokenize.

        Block 0 (coarsest, t=K-1, sos-only seed) is the generation root; the standard block-causal
        mask (block i attends to blocks 0..i) then makes AR generation run coarse->fine, i.e. blur
        weakens along the causal direction and each block attends to the more-blurred earlier steps.

        Targets are the *clean* low-pass token ids at each step's resolution (N of them), exactly
        as before. The per-step token grids are concatenated along the sequence dim; `Ls` records
        the per-block token count so the model can rebuild per-block position/time embeddings and a
        block-causal attention mask.

        Returns: x_BLCv (B, sum(Ls), Cvae) teacher-forcing input from the *blurred* images,
                 gt_BL (B, sum(Ls)) target ids from the *clean* low-pass images,
                 Ls (tuple of N ints) per-block token counts,
                 t_seq (N,) per-block timesteps.
        """
        var = self.var_wo_ddp
        B = inp_B3HW.shape[0]
        K = len(var.sigmas)

        # 1-2. random monotone (decreasing-blur) timestep schedule, shared across the batch;
        # the flip puts the forced v[-1]==1 endpoint at block 0 -> t==K-1 (fully blurred seed),
        # so block 0 is always the most-blurred coarsest step and blur weakens along the sequence.
        # kept on CPU; only python-int indices are used to index the GPU sigma/freq buffers.
        v = torch.randn(self.N).softmax(dim=0).cumsum(dim=0)        # (N,), increasing in (0,1]
        t_seq = (v * (K - 1)).round().long().flip(0)               # (N,), decreasing -> block 0 == K-1 (most blur)

        # DCT once; each step slices the low-frequency square it can represent.
        dct_full = DCT(inp_B3HW)
        xs, gts, Ls = [], [], []
        for i in range(self.N):
            t_i = int(t_seq[i])
            size = self._t_to_size(t_i)
            dct = dct_full[:, :, :size, :size]
            blur = (- var.sigmas[t_i] * var.freqs[:, :, :size, :size]).exp().to(dct)
            inp_blured = iDCT(blur * dct)
            inp_clean = iDCT(dct)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.vae_bf16):
                if self.fused_vae_encode:
                    both = self.vae_local.img_to_idxBl(torch.cat([inp_blured, inp_clean], dim=0))
                    idx_blured, idx_clean = both[:B], both[B:]
                else:
                    idx_blured = self.vae_local.img_to_idxBl(inp_blured)
                    idx_clean = self.vae_local.img_to_idxBl(inp_clean)
            xs.append(self.quantize_local.idxBl_to_var_input(idx_blured))   # (B, L_i, Cvae)
            gts.append(idx_clean)                                          # (B, L_i)
            Ls.append(idx_clean.shape[1])

        x_BLCv = torch.cat(xs, dim=1)
        gt_BL = torch.cat(gts, dim=1)
        return x_BLCv, gt_BL, tuple(Ls), t_seq

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            x_BLCv, gt_BL, Ls, t_seq = self.fetch_blur_tokens(inp_B3HW)
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv, t_seq, Ls)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data.reshape(-1, V), gt_BL.reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen],
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        # DCT blur curriculum + frozen-VAE tokenization (no grad; handled in fetch_blur_tokens)
        x_BLCv, gt_BL, Ls, t_seq = self.fetch_blur_tokens(inp_B3HW)

        with self.var_opt.amp_ctx:
            logits_BLV = self.var(label_B, x_BLCv, t_seq, Ls)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            # mean cross-entropy over all tokens of the N-step sequence
            loss = loss.mean()
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = self.val_loss(logits_BLV.data.reshape(-1, V), gt_BL.reshape(-1)).item()
            acc_tail = (pred_BL == gt_BL).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                pred, tar = logits_BLV.data.reshape(-1, V), gt_BL.reshape(-1)
                acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                ce = self.val_loss(pred, tar).item()
                wandb.log({f'AR_iter_loss/Acc': acc, f'AR_iter_loss/CE': ce}, step=g_it)
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'label_smooth': self.label_smooth,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
