
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseTimeSeriesDataset
from .preprocess import chronological_split, fit_transform_by_train, infer_feature_cols, split_contiguous_segments


class ForecastWindowDataset(BaseTimeSeriesDataset):
    def __init__(self, windows: dict[str, np.ndarray]):
        super().__init__(task='forecast')
        self.tensors = {}
        for k, v in windows.items():
            dtype = torch.long if k == 'y_cls' else torch.float32
            self.tensors[k] = torch.tensor(v, dtype=dtype)

    def __len__(self) -> int:
        return len(self.tensors['x'])

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.tensors.items()}


def _build_time_mark(seg: pd.DataFrame, time_col: Optional[str], length: int) -> np.ndarray:
    if time_col is None or time_col not in seg.columns:
        return np.zeros((length, 5), dtype=np.float32)
    dt = pd.to_datetime(seg[time_col])
    mark = np.stack([
        dt.dt.month.values,
        dt.dt.day.values,
        dt.dt.weekday.values,
        dt.dt.hour.values,
        dt.dt.minute.values,
    ], axis=-1)
    return mark.astype(np.float32)


def _build_forecast_windows(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str], endo_cols: list[str], exo_cols: list[str], seq_len: int, pred_len: int, stride: int, time_col: str | None, expected_freq: str | None):
    segments = [df]
    if time_col is not None and expected_freq is not None:
        segments = split_contiguous_segments(df, time_col=time_col, expected_delta=expected_freq)

    outs = {k: [] for k in ['x', 'y', 'endo_x', 'exo_x', 'x_mark', 'endo_mask', 'exo_mask']}
    for seg in segments:
        feat = seg[feature_cols].values.astype(np.float32)
        tgt = seg[target_cols].values.astype(np.float32)
        endo = seg[endo_cols].values.astype(np.float32)
        exo = seg[exo_cols].values.astype(np.float32) if exo_cols else np.zeros((len(seg), 0), dtype=np.float32)
        mark = _build_time_mark(seg, time_col, len(seg))
        total_need = seq_len + pred_len
        if len(seg) < total_need:
            continue
        for start in range(0, len(seg) - total_need + 1, stride):
            mid = start + seq_len
            end = mid + pred_len
            outs['x'].append(feat[start:mid])
            outs['y'].append(tgt[mid:end])
            outs['endo_x'].append(endo[start:mid])
            outs['exo_x'].append(exo[start:mid])
            outs['x_mark'].append(mark[start:mid])
            outs['endo_mask'].append(np.ones((seq_len, len(endo_cols)), dtype=np.float32))
            outs['exo_mask'].append(np.ones((seq_len, len(exo_cols)), dtype=np.float32) if exo_cols else np.zeros((seq_len, 0), dtype=np.float32))
    if not outs['x']:
        raise ValueError('No forecasting windows were created. Check seq_len/pred_len/frequency constraints.')
    return {k: np.stack(v) for k, v in outs.items()}


def build_forecast_loaders(config: dict):
    cfg = config['forecast']
    data_root = Path(config['paths']['data_root'])
    file_path = data_root / cfg['file_name']
    df = pd.read_csv(file_path)

    time_col = cfg.get('time_col')
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    target_cols = cfg['target_cols']
    feature_cols = cfg.get('feature_cols') or infer_feature_cols(df, time_col=time_col)
    endo_cols = cfg.get('endo_cols') or target_cols
    exo_cols = cfg.get('exo_cols')
    if exo_cols is None:
        exo_cols = [c for c in feature_cols if c not in endo_cols]

    train_df, val_df, test_df = chronological_split(df, cfg['train_ratio'], cfg['val_ratio'], cfg['test_ratio'])
    train_scaled, [val_scaled, test_scaled], scaler_bundle = fit_transform_by_train(train_df, [val_df, test_df], feature_cols, scaler_kind=cfg['normalize'])

    seq_len = cfg['seq_len']
    pred_len = cfg['pred_len']
    stride = cfg['stride']
    expected_freq = cfg.get('expected_freq')

    train_w = _build_forecast_windows(train_scaled, feature_cols, target_cols, endo_cols, exo_cols, seq_len, pred_len, stride, time_col, expected_freq)
    val_w = _build_forecast_windows(val_scaled, feature_cols, target_cols, endo_cols, exo_cols, seq_len, pred_len, stride, time_col, expected_freq)
    test_w = _build_forecast_windows(test_scaled, feature_cols, target_cols, endo_cols, exo_cols, seq_len, pred_len, stride, time_col, expected_freq)

    train_ds = ForecastWindowDataset(train_w)
    val_ds = ForecastWindowDataset(val_w)
    test_ds = ForecastWindowDataset(test_w)

    train_cfg = config['train']
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    metadata = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'endo_cols': endo_cols,
        'exo_cols': exo_cols,
        'input_dim': len(feature_cols),
        'output_dim': len(target_cols),
        'n_endo': len(endo_cols),
        'n_exo': len(exo_cols),
        'seq_len': seq_len,
        'pred_len': pred_len,
        'scaler': scaler_bundle,
        'raw_splits': {'train': train_df, 'val': val_df, 'test': test_df},
        'scaled_splits': {'train': train_scaled, 'val': val_scaled, 'test': test_scaled},
    }
    return train_loader, val_loader, test_loader, metadata
