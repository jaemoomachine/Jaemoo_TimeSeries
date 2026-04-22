
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import SharedTimeSeriesEncoder
from .heads import ClassificationHead, ForecastHead, ReconstructionHead
from .timexer_family import UnifiedTimeXerModel


class UnifiedTimeSeriesModel(nn.Module):
    def __init__(
        self,
        task: str,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pred_len: int | None = None,
        output_dim: int | None = None,
        num_classes: int | None = None,
        pooling: str = 'mean',
        model_name: str = 'transformer',
        seq_len: int | None = None,
        n_endo: int | None = None,
        n_exo: int | None = None,
        patch_len: int = 16,
        patch_stride: int = 16,
        patch_padding: int = 0,
        n_globaltokens: int = 1,
        attn_dropout: float = 0.0,
        revin_affine: bool = True,
        revin_subtract_last: bool = False,
        tau_hidden_dim: int = 64,
        tau_max: float = 20.0,
        delta_clip: float = 10.0,
    ):
        super().__init__()
        self.task = task
        self.model_name = model_name

        if model_name in {'timexer', 'timexer_revin', 'timexer_nst'}:
            if seq_len is None or n_endo is None or n_exo is None:
                raise ValueError('TimeXer family requires seq_len, n_endo, n_exo')
            self.model = UnifiedTimeXerModel(
                task=task,
                variant=model_name,
                seq_len=seq_len,
                n_endo=n_endo,
                n_exo=n_exo,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_dropout=attn_dropout,
                patch_len=patch_len,
                patch_stride=patch_stride,
                patch_padding=patch_padding,
                n_globaltokens=n_globaltokens,
                pred_len=pred_len,
                num_classes=num_classes,
                pooling=pooling,
                revin_affine=revin_affine,
                revin_subtract_last=revin_subtract_last,
                tau_hidden_dim=tau_hidden_dim,
                tau_max=tau_max,
                delta_clip=delta_clip,
            )
        else:
            self.encoder = SharedTimeSeriesEncoder(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            if task == 'forecast':
                if pred_len is None or output_dim is None:
                    raise ValueError('Forecast task requires pred_len and output_dim.')
                self.head = ForecastHead(d_model=d_model, pred_len=pred_len, output_dim=output_dim, mode='sequence_last')
            elif task == 'anomaly':
                if output_dim is None:
                    raise ValueError('Anomaly task requires output_dim.')
                self.head = ReconstructionHead(d_model=d_model, output_dim=output_dim, mode='sequence_projection')
            elif task == 'classification':
                if num_classes is None:
                    raise ValueError('Classification task requires num_classes.')
                self.head = ClassificationHead(d_model=d_model, num_classes=num_classes, pooling=pooling)
            else:
                raise ValueError(f'Unsupported task: {task}')

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if self.model_name in {'timexer', 'timexer_revin', 'timexer_nst'}:
            if not isinstance(batch, dict):
                raise ValueError('TimeXer family expects batch dict input')
            return self.model(batch)
        x = batch['x'] if isinstance(batch, dict) else batch
        h = self.encoder(x)
        return self.head(h)
