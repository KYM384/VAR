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

    @torch.no_grad()
    def fetch_blur_tokens(self, inp_B3HW: FTen):
        """Sample a per-sample blur timestep t ~ Uniform[0, K) and tokenize with the frozen VAE.

        Unlike the old chunk-first sampling (draw one size/chunk for the whole minibatch, then t),
        every image now gets an independent, fully uniform t. Sizes therefore differ within the
        batch, so tokens are right-padded to L_max = latent_size**2 with dummy tokens and we return
        a key-padding attention bias (plus a token_mask for the loss). The model then runs on a
        fixed L_max sequence while each image effectively keeps its own resolution.

        Returns:
            x_BLCv     (B, L_max, Cvae)  teacher-forcing input from the *blurred* image:
                                         continuous encoder features, quantization skipped
            gt_BL      (B, L_max)        target token ids from the *clean* low-pass image
                                         (dummy/padding positions = 0; masked out in the loss)
            t          (B,)             per-sample blur timestep (CPU long)
            token_mask (B, L_max) bool  True at real tokens, False at padding
            attn_bias  (B, 1, 1, L_max) additive mask: 0 at real keys, -inf at padding keys
        """
        var = self.var_wo_ddp
        B = inp_B3HW.shape[0]
        dev = inp_B3HW.device
        K = len(var.sigmas)
        L_max = self.L
        Cvae = self.vae_local.Cvae

        # completely uniform per-sample timestep (no chunk-first sampling -> no importance weight)
        t = torch.randint(0, K, (B,))
        size_per = [VAR._t_to_size(int(ti)) for ti in t.tolist()]

        x_BLCv = torch.zeros(B, L_max, Cvae, device=dev)
        gt_BL = torch.zeros(B, L_max, dtype=torch.long, device=dev)
        token_mask = torch.zeros(B, L_max, dtype=torch.bool, device=dev)

        # one batched VAE encode per distinct size present in the minibatch
        for size in sorted(set(size_per), reverse=True):
            sel = [i for i, s in enumerate(size_per) if s == size]
            Lg = (size // 16) ** 2
            sub_inp = inp_B3HW[sel]
            sub_sigma = var.sigmas[t[sel].to(dev)].reshape(-1, 1, 1, 1)
            dct = DCT(sub_inp)[:, :, :size, :size]
            blur = (- sub_sigma * var.freqs[:, :, :size, :size]).exp().to(dct)
            # amplitude-preserving low-pass: size/256 undoes the DCT-crop amplification so the
            # frozen VAE sees in-range [-1,1] images (same scaling as autoregressive_infer_cfg).
            scale = size / 256
            blured = iDCT(blur * dct) * scale
            clean = iDCT(dct) * scale
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.vae_bf16):
                if self.fused_vae_encode:
                    f_both = self.vae_local.img_to_f(torch.cat([blured, clean], dim=0))
                    f_blur, f_clean = f_both[:len(sel)], f_both[len(sel):]
                else:
                    f_blur = self.vae_local.img_to_f(blured)
                    f_clean = self.vae_local.img_to_f(clean)
                # targets stay quantized (CE over the codebook); only the input skips quantization
                gt_sub = self.quantize_local.f_to_idxBl_or_fhat(f_clean, to_fhat=False)
            x_sub = f_blur.permute(0, 2, 3, 1).reshape(len(sel), Lg, Cvae)   # (len(sel), Lg, Cvae) continuous
            x_BLCv[sel, :Lg] = x_sub.to(x_BLCv.dtype)
            gt_BL[sel, :Lg] = gt_sub
            token_mask[sel, :Lg] = True

        # key-padding additive bias: real keys -> 0, dummy keys -> -inf (broadcasts over heads/queries)
        attn_bias = torch.zeros(B, 1, 1, L_max, device=dev)
        attn_bias.masked_fill_(~token_mask.view(B, 1, 1, L_max), float('-inf'))
        return x_BLCv, gt_BL, t, token_mask, attn_bias

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        ce_sum, correct_sum, token_sum = 0.0, 0.0, 0.0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            x_BLCv, gt_BL, t, token_mask, attn_bias = self.fetch_blur_tokens(inp_B3HW)
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv, t.to(x_BLCv), attn_bias)

            # sizes vary per sample, so reduce over *valid* (non-padding) tokens only
            valid = token_mask.view(-1)
            logits_flat = logits_BLV.data.view(-1, V)[valid]
            gt_flat = gt_BL.view(-1)[valid]
            nvalid = logits_flat.shape[0]
            ce_sum += self.val_loss(logits_flat, gt_flat).item() * nvalid
            correct_sum += (logits_flat.argmax(dim=-1) == gt_flat).sum().item()
            token_sum += nvalid
            tot += B
        self.var_wo_ddp.train(training)

        # fp64: correct_sum/token_sum are token counts that can exceed fp32's exact-int range (2^24)
        stats = torch.tensor([ce_sum, correct_sum, token_sum, float(tot)], device=dist.get_device(), dtype=torch.float64)
        dist.allreduce(stats)
        ce_sum, correct_sum, token_sum, tot = stats.tolist()
        token_sum = max(token_sum, 1.0)
        L_mean = ce_sum / token_sum
        acc_mean = correct_sum / token_sum * 100
        # no separate coarse/tail scale in this single-prediction setup: report the same value twice
        return L_mean, L_mean, acc_mean, acc_mean, round(tot), time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen],
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        # DCT blur curriculum + frozen-VAE tokenization (no grad; handled in fetch_blur_tokens).
        # Uniform per-sample t -> mixed sizes -> padded tokens + token_mask/attn_bias.
        x_BLCv, gt_BL, t, token_mask, attn_bias = self.fetch_blur_tokens(inp_B3HW)

        with self.var_opt.amp_ctx:
            logits_BLV = self.var(label_B, x_BLCv, t.to(x_BLCv), attn_bias)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            # mean over *real* tokens only (padding is masked out); uniform t needs no reweighting
            loss = (loss * token_mask).sum() / token_mask.sum().clamp_min(1)

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log (metrics over valid/non-padding tokens only)
        valid = token_mask.view(-1)
        pred_flat = logits_BLV.data.view(-1, V).argmax(dim=-1)
        gt_flat = gt_BL.view(-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V)[valid], gt_flat[valid]).item()
            acc_mean = (pred_flat[valid] == gt_flat[valid]).float().mean().item() * 100
            Ltail = Lmean
            acc_tail = acc_mean
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_flat[valid].bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                pred, tar = logits_BLV.data.view(-1, V)[valid], gt_flat[valid]
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
