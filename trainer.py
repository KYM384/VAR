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
        """Apply the DCT frequency-blur curriculum and tokenize with the frozen VAE.

        Shared by train_step and eval_ep so the (previously copy-pasted) logic stays in sync.
        Returns: x_BLCv_wo_first_l (teacher-forcing input from the *blurred* image),
                 gt_BL (target token ids from the *clean* low-pass image), t, weight.
        The torch CPU RNG draw order (t_chunk, then t) is preserved exactly, so the random
        blur schedule is unchanged from the original code.
        """
        var = self.var_wo_ddp
        B = inp_B3HW.shape[0]
        dev = inp_B3HW.device

        t_chunk = torch.randint(0, 5, (1,)).item()
        t_min = 90 + (180-90)//3 * (t_chunk-1) if t_chunk > 0 else 0
        t_max = 90 + (180-90)//3 * (t_chunk) if t_chunk < 4 else len(var.sigmas)
        t = torch.randint(t_min, t_max, (B,))
        size = 256 // 2 ** int(t_chunk)
        weight = (t_max - t_min) / ((180-90)//3)

        # var.sigmas / var.freqs are now GPU buffers; t is kept on CPU (to preserve the RNG
        # stream) and moved to the buffer's device only for indexing.
        dct_B3HW = DCT(inp_B3HW)[:, :, :size, :size]
        blur = (- var.sigmas[t.to(dev)].reshape(-1, 1, 1, 1) * var.freqs[:, :, :size, :size]).exp().to(dct_B3HW)
        inp_B3HW_blured = iDCT(blur * dct_B3HW)
        inp_B3HW_clean = iDCT(dct_B3HW)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.vae_bf16):
            if self.fused_vae_encode:
                both = self.vae_local.img_to_idxBl(torch.cat([inp_B3HW_blured, inp_B3HW_clean], dim=0))
                gt_idx_Bl, gt_BL = both[:B], both[B:]
            else:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW_blured)
                gt_BL = self.vae_local.img_to_idxBl(inp_B3HW_clean)
        x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        return x_BLCv_wo_first_l, gt_BL, t, weight

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

            x_BLCv_wo_first_l, gt_BL, t, weight = self.fetch_blur_tokens(inp_B3HW)
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l, t.to(x_BLCv_wo_first_l))
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B * weight
            L_tail += self.val_loss(logits_BLV.data.reshape(-1, V), gt_BL.reshape(-1)) * B * weight
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100)
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
        x_BLCv_wo_first_l, gt_BL, t, weight = self.fetch_blur_tokens(inp_B3HW)

        with self.var_opt.amp_ctx:
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l, t.to(x_BLCv_wo_first_l))
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            # lw = self.loss_weight
            # loss = loss.mul(lw).sum(dim=-1).mean()
            loss = weight * loss.mean()
        
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
