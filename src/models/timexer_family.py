
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ForecastHead, ReconstructionHead, ClassificationHead
from .backbone import PositionalEncoding

def _to_bnl_mask(mask: Optional[torch.Tensor], x_bln: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    # accept [B,L,N] or [B,N,L]
    if mask.dim() != 3:
        raise ValueError(f"mask must be 3D, got {mask.shape}")
    if mask.shape == x_bln.shape:
        return (mask > 0).to(torch.uint8)
    if mask.shape == (x_bln.shape[0], x_bln.shape[2], x_bln.shape[1]):
        return (mask.permute(0, 2, 1).contiguous() > 0).to(torch.uint8)
    raise ValueError(f"mask shape {mask.shape} incompatible with x shape {x_bln.shape}")


def _key_padding_from_keep(keep: torch.Tensor) -> torch.Tensor:
    if keep.dtype != torch.bool:
        keep = keep > 0
    return ~keep


def _ensure_any_kept(keep: torch.Tensor) -> torch.Tensor:
    if keep.dtype != torch.bool:
        keep = keep > 0
    keep = keep.clone()
    zero = keep.sum(dim=1) == 0
    if zero.any():
        keep[zero, -1] = True
    return keep


def _expand_for_vars(x_bsd: Optional[torch.Tensor], n_vars: int) -> Optional[torch.Tensor]:
    if x_bsd is None:
        return None
    B, S, D = x_bsd.shape
    return x_bsd.unsqueeze(1).expand(B, n_vars, S, D).reshape(B * n_vars, S, D)


def _expand_keep_for_vars(keep_bs: Optional[torch.Tensor], n_vars: int) -> Optional[torch.Tensor]:
    if keep_bs is None:
        return None
    B, S = keep_bs.shape
    return keep_bs.unsqueeze(1).expand(B, n_vars, S).reshape(B * n_vars, S)


def _safe_time_mark(x_mark: Optional[torch.Tensor], B: int, L: int, device=None) -> torch.Tensor:
    if x_mark is None:
        return torch.zeros(B, L, 5, device=device)
    if x_mark.dim() != 3 or x_mark.shape[0] != B or x_mark.shape[1] != L:
        raise ValueError(f"x_mark must be [B,L,5]-like, got {x_mark.shape}")
    if x_mark.shape[2] < 5:
        pad = torch.zeros(B, L, 5 - x_mark.shape[2], device=x_mark.device, dtype=x_mark.dtype)
        x_mark = torch.cat([x_mark, pad], dim=-1)
    return x_mark[..., :5]


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int, keepdim: bool = True) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    m = mask.to(x.dtype)
    denom = m.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (x * m).sum(dim=dim, keepdim=keepdim) / denom



class EndoPatchEmbed(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1, padding: int = 0):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding = int(padding)
        self.value_proj = nn.Linear(self.patch_len, d_model)
        self.time_proj = nn.Linear(5, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def _patchify(self, x_bnl: torch.Tensor) -> torch.Tensor:
        # x: [B,N,L] -> [B,N,P,patch_len]
        if self.padding > 0:
            x_bnl = F.pad(x_bnl, (0, self.padding))
        patches = x_bnl.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches.contiguous()

    def _patchify_mark(self, x_mark: torch.Tensor, total_len: int) -> torch.Tensor:
        # x_mark [B,L,5] -> [B,P,5]
        if self.padding > 0:
            x_mark_t = x_mark.transpose(1, 2)
            x_mark_t = F.pad(x_mark_t, (0, self.padding))
            x_mark = x_mark_t.transpose(1, 2)
        patches = x_mark.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B,P,5,patch]
        if patches.dim() != 4:
            raise RuntimeError(f"unexpected mark patches shape {patches.shape}")
        patches = patches.permute(0, 1, 3, 2).contiguous()  # [B,P,patch,5]
        return patches.mean(dim=2)

    def forward(self, x_bln: torch.Tensor, x_mark: Optional[torch.Tensor], input_mask: Optional[torch.Tensor]):
        B, L, N = x_bln.shape
        x_bnl = x_bln.permute(0, 2, 1).contiguous()
        mask_bln = _to_bnl_mask(input_mask, x_bln)
        mask_bnl = None if mask_bln is None else mask_bln.permute(0, 2, 1).contiguous()

        patches = self._patchify(x_bnl)  # [B,N,P,patch]
        B2, N2, P, PL = patches.shape
        assert B2 == B and N2 == N

        if mask_bnl is None:
            patch_keep = torch.ones(B, N, P, device=x_bln.device, dtype=torch.bool)
        else:
            patch_mask = self._patchify(mask_bnl.to(torch.float32))
            patch_keep = patch_mask.sum(dim=-1) > 0

        mark = _safe_time_mark(x_mark, B=B, L=L, device=x_bln.device)
        mark_patch = self._patchify_mark(mark, total_len=L)  # [B,P,5]
        time_emb = self.time_proj(mark_patch.float())
        time_emb = time_emb.unsqueeze(1).expand(B, N, P, -1)

        tok = self.value_proj(patches.float()) + time_emb
        tok = tok.view(B * N, P, -1)
        tok = self.pos(tok)
        tok = self.dropout(tok)

        patch_keep_bnP = patch_keep.view(B * N, P)
        tok = tok * patch_keep_bnP.unsqueeze(-1).to(tok.dtype)
        meta = {'B': B, 'N': N, 'P': P, 'patch_len': self.patch_len, 'stride': self.stride, 'padding': self.padding}
        return tok, patch_keep_bnP, patch_keep, meta


class ExoEmbeddingInverted(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.value_proj = nn.Linear(self.seq_len, d_model, bias=False)
        self.time_proj = nn.Linear(5, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, exo_bln: Optional[torch.Tensor], x_mark: Optional[torch.Tensor], exo_mask: Optional[torch.Tensor]):
        if exo_bln is None:
            return None, None, None
        B, L, N = exo_bln.shape
        if N == 0:
            return None, None, None
        x_bnl = exo_bln.permute(0, 2, 1).contiguous()
        tok = self.value_proj(x_bnl.float())
        mark = _safe_time_mark(x_mark, B=B, L=L, device=exo_bln.device)
        time_ctx = self.time_proj(mark.float().mean(dim=1)).unsqueeze(1)
        tok = tok + time_ctx
        tok = self.dropout(tok)

        m = _to_bnl_mask(exo_mask, exo_bln)
        if m is None:
            keep = torch.ones(B, N, device=exo_bln.device, dtype=torch.bool)
        else:
            keep = m.permute(0, 2, 1).any(dim=-1)
        tok = tok * keep.unsqueeze(-1).to(tok.dtype)
        return tok, keep, {'exo_token_keep': keep}


class GlobalTokenBank(nn.Module):
    def __init__(self, d_model: int, n_globaltokens: int):
        super().__init__()
        self.n_globaltokens = int(n_globaltokens)
        self.glb = nn.Parameter(torch.randn(1, 1, self.n_globaltokens, d_model) * 0.02)

    def forward(self, B: int, N: int) -> torch.Tensor:
        return self.glb.expand(B, N, self.n_globaltokens, -1)

@dataclass
class NonStationaryNormalizer:
    eps: float = 1e-5

    def normalize(self, x_bnl: torch.Tensor, mask_bnl: Optional[torch.Tensor] = None):
        mean = _masked_mean(x_bnl, mask_bnl, dim=-1, keepdim=True).detach()
        xc = x_bnl - mean
        if mask_bnl is None:
            var = torch.var(xc, dim=-1, keepdim=True, unbiased=False)
        else:
            m = mask_bnl.to(x_bnl.dtype)
            var = ((xc * m) ** 2).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        std = torch.sqrt(var + self.eps).detach()
        return xc / std, {'mean': mean, 'std': std}

    def denormalize(self, y: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return y * state['std'] + state['mean']


class RevIN(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, n_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, n_channels, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x_bnl: torch.Tensor, mask_bnl: Optional[torch.Tensor] = None):
        if self.subtract_last:
            ref = x_bnl[..., -1:].detach()
        else:
            ref = _masked_mean(x_bnl, mask_bnl, dim=-1, keepdim=True).detach()
        xc = x_bnl - ref
        if mask_bnl is None:
            var = torch.var(xc, dim=-1, keepdim=True, unbiased=False)
        else:
            m = mask_bnl.to(x_bnl.dtype)
            var = ((xc * m) ** 2).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        std = torch.sqrt(var + self.eps).detach().clamp_min(1e-4)
        y = xc / std
        if self.affine:
            y = y * self.gamma + self.beta
        return y, {'ref': ref, 'std': std}

    def denorm(self, y: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.affine:
            y = (y - self.beta) / (self.gamma.clamp_min(1e-4))
        return y * state['std'] + state['ref']


class StandardAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, q, k, v, key_padding_mask=None, tau=None, delta=None):
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return _finite(out), None


class DSAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, key_padding_mask=None, tau=None, delta=None):
        B, Tq, D = q.shape
        Tk = k.size(1)
        q = self.q_proj(q).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        if tau is not None:
            if tau.dim() == 1:
                tau = tau.view(B, 1, 1, 1)
            elif tau.dim() == 2:
                tau = tau.view(B, 1, 1, 1)
            q = q * tau.to(q.dtype)

        attn_mask = None
        if delta is not None or key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, Tk), device=q.device, dtype=q.dtype)
            if delta is not None:
                attn_mask = attn_mask + delta.view(B, 1, 1, Tk).to(q.dtype)
            if key_padding_mask is not None:
                neg = -1e4 if q.dtype in (torch.float16, torch.bfloat16) else -1e9
                all_masked = key_padding_mask.all(dim=1)
                if all_masked.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_masked, -1] = False
                attn_mask = attn_mask.masked_fill(key_padding_mask.view(B, 1, 1, Tk), neg)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training and self.dropout > 0 else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.o_proj(out)
        return _finite(out), None


# =========================
# Encoder blocks
# =========================

class TimeXerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float, n_globaltokens: int, attention_cls=StandardAttentionLayer):
        super().__init__()
        self.n_globaltokens = int(n_globaltokens)
        self.self_attn = attention_cls(d_model, n_heads, attn_dropout)
        self.cross_attn = attention_cls(d_model, n_heads, attn_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, endo_tokens, exo_tokens, self_kpm, cross_kpm, endo_query_keep, glb_query_keep, tau=None, delta_self=None, delta_cross=None):
        sa_out, _ = self.self_attn(endo_tokens, endo_tokens, endo_tokens, key_padding_mask=self_kpm, tau=tau, delta=delta_self)
        x = self.norm1(endo_tokens + self.dropout(sa_out))
        if endo_query_keep is not None:
            x = x * endo_query_keep.unsqueeze(-1).to(x.dtype)

        T = x.size(1)
        G = self.n_globaltokens
        patch = x[:, : T - G, :]
        glb = x[:, T - G :, :]

        if exo_tokens is not None and exo_tokens.size(1) > 0:
            ca_out, _ = self.cross_attn(glb, exo_tokens, exo_tokens, key_padding_mask=cross_kpm, tau=tau, delta=delta_cross)
            glb = self.norm2(glb + self.dropout(ca_out))
        else:
            glb = self.norm2(glb)

        if glb_query_keep is not None:
            glb = glb * glb_query_keep.unsqueeze(-1).to(glb.dtype)

        x = torch.cat([patch, glb], dim=1)
        y = self.ffn(x)
        x = self.norm3(x + self.dropout(y))
        if endo_query_keep is not None:
            x = x * endo_query_keep.unsqueeze(-1).to(x.dtype)
        return _finite(x)


class NSTProjector(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_bnl: torch.Tensor, mean_bnl: torch.Tensor, std_bnl: torch.Tensor) -> torch.Tensor:
        B, N, L = x_bnl.shape
        z = x_bnl.reshape(B * N, 1, L)
        z = self.pool(F.gelu(self.conv(z))).squeeze(-1)
        stats = torch.cat([mean_bnl.reshape(B * N, 1), std_bnl.reshape(B * N, 1)], dim=-1)
        return self.mlp(torch.cat([z, stats], dim=-1))


class _BaseTimeXerBackbone(nn.Module):
    def __init__(self, seq_len: int, n_endo: int, n_exo: int, d_model: int, n_heads: int, e_layers: int, d_ff: int, patch_len: int, patch_stride: int, padding: int, n_globaltokens: int, dropout: float, attn_dropout: float, attention_cls=StandardAttentionLayer):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_endo = int(n_endo)
        self.n_exo = int(n_exo)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.padding = int(padding)
        self.n_globaltokens = int(n_globaltokens)

        self.endo_patch = EndoPatchEmbed(patch_len, patch_stride, d_model, dropout=dropout, padding=padding)
        self.exo_embed = ExoEmbeddingInverted(seq_len, d_model, dropout=dropout)
        self.glb = GlobalTokenBank(d_model, n_globaltokens)
        self.blocks = nn.ModuleList([
            TimeXerEncoderBlock(d_model, n_heads, d_ff, dropout, attn_dropout, n_globaltokens, attention_cls=attention_cls)
            for _ in range(e_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def _normalize(self, endo_bnl: torch.Tensor, mask_bnl: Optional[torch.Tensor]):
        return endo_bnl, {}

    def denormalize(self, y_bnl: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return y_bnl

    def _build_attention_bias(self, x_norm_bnl: torch.Tensor, state: Dict[str, torch.Tensor], token_num: int, exo_tokens_len: int):
        return None, None, None

    def forward(self, endo_x: torch.Tensor, exo_x: Optional[torch.Tensor], x_mark: Optional[torch.Tensor], endo_mask: Optional[torch.Tensor], exo_mask: Optional[torch.Tensor]):
        # inputs [B,L,N]
        B, L, N = endo_x.shape
        if N != self.n_endo:
            raise ValueError(f"expected n_endo={self.n_endo}, got {N}")
        x_mark = _safe_time_mark(x_mark, B, L, device=endo_x.device)
        endo_mask_bln = _to_bnl_mask(endo_mask, endo_x)
        endo_mask_bnl = None if endo_mask_bln is None else endo_mask_bln.permute(0, 2, 1).contiguous()

        endo_bnl = endo_x.permute(0, 2, 1).contiguous()
        x_norm_bnl, norm_state = self._normalize(endo_bnl, endo_mask_bnl)
        x_norm_bln = x_norm_bnl.permute(0, 2, 1).contiguous()

        endo_emb, patch_keep_bnP, patch_keep_bnp, meta = self.endo_patch(x_norm_bln, x_mark, endo_mask_bln)
        P = meta['P']
        glb = self.glb(B, N).reshape(B * N, self.n_globaltokens, self.d_model)
        endo_tokens = torch.cat([endo_emb, glb], dim=1)
        token_keep = torch.cat([patch_keep_bnP, torch.ones(B * N, self.n_globaltokens, device=endo_emb.device, dtype=torch.bool)], dim=1)
        token_keep = _ensure_any_kept(token_keep)
        self_kpm = _key_padding_from_keep(token_keep)
        glb_keep = torch.ones(B * N, self.n_globaltokens, device=endo_emb.device, dtype=torch.bool)

        exo_tokens_bsd, exo_keep_bs, _ = self.exo_embed(exo_x, x_mark, exo_mask)
        exo_tokens = _expand_for_vars(exo_tokens_bsd, n_vars=N)
        exo_keep = _expand_keep_for_vars(exo_keep_bs, n_vars=N)
        if exo_keep is not None:
            exo_keep = _ensure_any_kept(exo_keep)
            cross_kpm = _key_padding_from_keep(exo_keep)
        else:
            cross_kpm = None

        token_num = P + self.n_globaltokens
        exo_len = 0 if exo_tokens_bsd is None else exo_tokens_bsd.size(1)
        tau, delta_self, delta_cross = self._build_attention_bias(x_norm_bnl, norm_state, token_num, exo_len)

        x = endo_tokens
        for blk in self.blocks:
            x = blk(x, exo_tokens, self_kpm, cross_kpm, token_keep, glb_keep, tau=tau, delta_self=delta_self, delta_cross=delta_cross)
        x = self.final_norm(x)
        x_bntd = x.view(B, N, token_num, self.d_model)
        patch_tokens = x_bntd[:, :, :P, :]
        global_tokens = x_bntd[:, :, P:, :]
        return {
            'tokens': x_bntd,
            'patch_tokens': patch_tokens,
            'global_tokens': global_tokens,
            'patch_keep': patch_keep_bnp,
            'meta': meta,
            'norm_state': norm_state,
        }


class TimeXerBackbone(_BaseTimeXerBackbone):
    def __init__(self, **kwargs):
        super().__init__(attention_cls=StandardAttentionLayer, **kwargs)
        self.normalizer = NonStationaryNormalizer()

    def _normalize(self, endo_bnl: torch.Tensor, mask_bnl: Optional[torch.Tensor]):
        return self.normalizer.normalize(endo_bnl, mask_bnl)

    def denormalize(self, y_bnl: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.normalizer.denormalize(y_bnl, state)


class TimeXerRevINBackbone(_BaseTimeXerBackbone):
    def __init__(self, revin_affine: bool = True, revin_subtract_last: bool = False, **kwargs):
        super().__init__(attention_cls=StandardAttentionLayer, **kwargs)
        self.revin = RevIN(n_channels=self.n_endo, affine=revin_affine, subtract_last=revin_subtract_last)

    def _normalize(self, endo_bnl: torch.Tensor, mask_bnl: Optional[torch.Tensor]):
        return self.revin.norm(endo_bnl, mask_bnl)

    def denormalize(self, y_bnl: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.revin.denorm(y_bnl, state)


class TimeXerNSTBackbone(_BaseTimeXerBackbone):
    def __init__(self, tau_hidden_dim: int = 64, tau_max: float = 20.0, delta_clip: float = 10.0, **kwargs):
        super().__init__(attention_cls=DSAttentionLayer, **kwargs)
        self.normalizer = NonStationaryNormalizer()
        self.tau_proj = NSTProjector(self.seq_len, tau_hidden_dim, 1)
        self.delta_self_proj = NSTProjector(self.seq_len, tau_hidden_dim, self._token_num())
        self.delta_cross_proj = NSTProjector(self.seq_len, tau_hidden_dim, max(self.n_exo, 1))
        self.tau_max = float(tau_max)
        self.delta_clip = float(delta_clip)

    def _token_num(self):
        Lpad = self.seq_len + self.padding
        patch_num = (Lpad - self.patch_len) // self.patch_stride + 1
        return patch_num + self.n_globaltokens

    def _normalize(self, endo_bnl: torch.Tensor, mask_bnl: Optional[torch.Tensor]):
        return self.normalizer.normalize(endo_bnl, mask_bnl)

    def denormalize(self, y_bnl: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.normalizer.denormalize(y_bnl, state)

    def _build_attention_bias(self, x_norm_bnl: torch.Tensor, state: Dict[str, torch.Tensor], token_num: int, exo_tokens_len: int):
        mean = state['mean']
        std = state['std']
        tau = F.softplus(self.tau_proj(x_norm_bnl.detach(), mean.detach(), std.detach())).squeeze(-1).clamp(1e-3, self.tau_max)
        delta_self = self.delta_self_proj(x_norm_bnl.detach(), mean.detach(), std.detach()).clamp(-self.delta_clip, self.delta_clip)
        if exo_tokens_len > 0:
            delta_cross = self.delta_cross_proj(x_norm_bnl.detach(), mean.detach(), std.detach())[:, :exo_tokens_len].clamp(-self.delta_clip, self.delta_clip)
        else:
            delta_cross = None
        return tau, delta_self, delta_cross


class UnifiedTimeXerModel(nn.Module):
    def __init__(self, task: str, variant: str, seq_len: int, n_endo: int, n_exo: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, attn_dropout: float = 0.0, patch_len: int = 16, patch_stride: int = 16, patch_padding: int = 0, n_globaltokens: int = 1, pred_len: Optional[int] = None, num_classes: Optional[int] = None, pooling: str = 'tokens_mean', revin_affine: bool = True, revin_subtract_last: bool = False, tau_hidden_dim: int = 64, tau_max: float = 20.0, delta_clip: float = 10.0):
        super().__init__()
        self.task = task
        self.variant = variant
        backbone_kwargs = dict(
            seq_len=seq_len,
            n_endo=n_endo,
            n_exo=n_exo,
            d_model=d_model,
            n_heads=nhead,
            e_layers=num_layers,
            d_ff=dim_feedforward,
            patch_len=patch_len,
            patch_stride=patch_stride,
            padding=patch_padding,
            n_globaltokens=n_globaltokens,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        if variant == 'timexer':
            self.backbone = TimeXerBackbone(**backbone_kwargs)
        elif variant == 'timexer_revin':
            self.backbone = TimeXerRevINBackbone(revin_affine=revin_affine, revin_subtract_last=revin_subtract_last, **backbone_kwargs)
        elif variant == 'timexer_nst':
            self.backbone = TimeXerNSTBackbone(tau_hidden_dim=tau_hidden_dim, tau_max=tau_max, delta_clip=delta_clip, **backbone_kwargs)
        else:
            raise ValueError(f'Unsupported TimeXer variant: {variant}')

        token_num = ((seq_len + patch_padding - patch_len) // patch_stride + 1) + n_globaltokens
        if task == 'forecast':
            if pred_len is None:
                raise ValueError('pred_len is required for forecast task')
            self.head = ForecastHead(d_model=d_model, token_num=token_num, pred_len=pred_len, dropout=dropout, mode='token_flatten')
        elif task == 'anomaly':
            self.head = ReconstructionHead(d_model=d_model, patch_len=patch_len, stride=patch_stride, seq_len=seq_len, mode='patch_overlap')
        elif task == 'classification':
            if num_classes is None:
                raise ValueError('num_classes is required for classification task')
            self.head = ClassificationHead(d_model=d_model, num_classes=num_classes, pooling=pooling, dropout=dropout)
        else:
            raise ValueError(task)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.backbone(
            endo_x=batch['endo_x'],
            exo_x=batch.get('exo_x'),
            x_mark=batch.get('x_mark'),
            endo_mask=batch.get('endo_mask'),
            exo_mask=batch.get('exo_mask'),
        )
        if self.task == 'forecast':
            pred_bnp = self.head(features)
            pred_bnp = self.backbone.denormalize(pred_bnp, features['norm_state'])
            return pred_bnp.permute(0, 2, 1).contiguous()
        if self.task == 'anomaly':
            recon_bnl = self.head(features)
            recon_bnl = self.backbone.denormalize(recon_bnl, features['norm_state'])
            return recon_bnl.permute(0, 2, 1).contiguous()
        return self.head(features)
