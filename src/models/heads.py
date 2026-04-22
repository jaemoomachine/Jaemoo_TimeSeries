from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim, keepdim: bool = False) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    w = mask.to(x.dtype)
    num = (x * w).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return num / den


def _pool_sequence(h: torch.Tensor, pooling: str = 'mean') -> torch.Tensor:
    if pooling == 'mean':
        return h.mean(dim=1)
    if pooling == 'last':
        return h[:, -1, :]
    raise ValueError(f'Unsupported sequence pooling type: {pooling}')


def _pool_timexer_features(features: Dict[str, Any], pooling: str = 'tokens_mean') -> torch.Tensor:
    tokens = features['tokens']           # [B,N,T,D]
    patch_tokens = features.get('patch_tokens')
    global_tokens = features.get('global_tokens')
    patch_keep = features.get('patch_keep')

    if pooling in {'tokens_mean', 'all_tokens_mean', 'mean'}:
        return tokens.mean(dim=(1, 2))

    if pooling == 'tokens_last':
        return tokens[:, :, -1, :].mean(dim=1)

    if pooling in {'global', 'global_mean'}:
        if global_tokens is not None and global_tokens.numel() > 0:
            return global_tokens.mean(dim=(1, 2))
        return tokens.mean(dim=(1, 2))

    if pooling == 'patch_mean':
        if patch_tokens is None:
            return tokens.mean(dim=(1, 2))
        if patch_keep is None:
            return patch_tokens.mean(dim=(1, 2))
        mask = patch_keep.unsqueeze(-1).to(patch_tokens.dtype)   # [B,N,P,1]
        return _masked_mean(patch_tokens, mask, dim=(1, 2), keepdim=False)

    if pooling == 'patch_global_mean':
        if patch_tokens is None or patch_tokens.numel() == 0:
            return tokens.mean(dim=(1, 2))
        parts = [patch_tokens.mean(dim=(1, 2))]
        if global_tokens is not None and global_tokens.numel() > 0:
            parts.append(global_tokens.mean(dim=(1, 2)))
        return torch.stack(parts, dim=0).mean(dim=0)

    raise ValueError(f'Unsupported TimeXer pooling type: {pooling}')


class ForecastHead(nn.Module):
    def __init__(self, d_model: int, pred_len: int, output_dim: int | None = None, token_num: int | None = None, dropout: float = 0.0, mode: str = 'sequence_last'):
        super().__init__()
        self.pred_len = int(pred_len)
        self.output_dim = None if output_dim is None else int(output_dim)
        self.token_num = None if token_num is None else int(token_num)
        self.mode = mode
        self.dropout = nn.Dropout(float(dropout))

        if self.mode == 'sequence_last':
            if self.output_dim is None:
                raise ValueError('output_dim is required for sequence_last forecast head')
            self.proj = nn.Linear(d_model, self.pred_len * self.output_dim)
        elif self.mode == 'token_flatten':
            if self.token_num is None:
                raise ValueError('token_num is required for token_flatten forecast head')
            self.flatten = nn.Flatten(start_dim=-2)
            self.proj = nn.Linear(d_model * self.token_num, self.pred_len)
        else:
            raise ValueError(f'Unsupported forecast head mode: {mode}')

    def forward(self, x: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if self.mode == 'sequence_last':
            if isinstance(x, dict):
                raise ValueError('sequence_last forecast head expects tensor input')
            last = x[:, -1, :]
            out = self.dropout(self.proj(last))
            return out.view(x.size(0), self.pred_len, self.output_dim)

        if not isinstance(x, dict):
            raise ValueError('token_flatten forecast head expects feature dict input')
        tokens = x['tokens']  # [B,N,T,D]
        z = self.flatten(tokens.permute(0, 1, 3, 2).contiguous())
        z = self.dropout(self.proj(z))
        return z  # [B,N,pred_len]


class ReconstructionHead(nn.Module):
    def __init__(self, d_model: int, output_dim: int | None = None, patch_len: int | None = None, stride: int | None = None, seq_len: int | None = None, mode: str = 'sequence_projection'):
        super().__init__()
        self.mode = mode
        if self.mode == 'sequence_projection':
            if output_dim is None:
                raise ValueError('output_dim is required for sequence_projection reconstruction head')
            self.proj = nn.Linear(d_model, int(output_dim))
        elif self.mode == 'patch_overlap':
            if patch_len is None or stride is None or seq_len is None:
                raise ValueError('patch_len, stride, seq_len are required for patch_overlap reconstruction head')
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            self.seq_len = int(seq_len)
            self.proj = nn.Linear(d_model, self.patch_len)
        else:
            raise ValueError(f'Unsupported reconstruction head mode: {mode}')

    def forward(self, x: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if self.mode == 'sequence_projection':
            if isinstance(x, dict):
                raise ValueError('sequence_projection reconstruction head expects tensor input')
            return self.proj(x)

        if not isinstance(x, dict):
            raise ValueError('patch_overlap reconstruction head expects feature dict input')
        patch_tokens = x['patch_tokens']
        patch_keep = x.get('patch_keep')
        B, N, P, _ = patch_tokens.shape
        patch_vals = self.proj(patch_tokens)  # [B,N,P,patch_len]
        if patch_keep is not None:
            patch_vals = patch_vals * patch_keep.unsqueeze(-1).to(patch_vals.dtype)

        total_len = (P - 1) * self.stride + self.patch_len
        recon = torch.zeros(B, N, total_len, device=patch_vals.device, dtype=patch_vals.dtype)
        counts = torch.zeros(B, N, total_len, device=patch_vals.device, dtype=patch_vals.dtype)
        for p in range(P):
            s = p * self.stride
            e = s + self.patch_len
            recon[:, :, s:e] += patch_vals[:, :, p, :]
            counts[:, :, s:e] += 1.0
        recon = recon / counts.clamp_min(1.0)
        return recon[:, :, : self.seq_len]


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, pooling: str = 'mean', dropout: float = 0.1):
        super().__init__()
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, int(num_classes)),
        )

    def forward(self, x: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if isinstance(x, dict):
            pooled = _pool_timexer_features(x, pooling=self.pooling)
        else:
            pooled = _pool_sequence(x, pooling=self.pooling)
        return self.net(pooled)
