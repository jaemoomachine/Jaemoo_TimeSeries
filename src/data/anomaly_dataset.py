
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseTimeSeriesDataset
from .preprocess import build_scaler, infer_feature_cols


class AnomalyWindowDataset(BaseTimeSeriesDataset):
    def __init__(self, windows: dict[str, np.ndarray]):
        super().__init__(task='anomaly')
        self.tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in windows.items()}

    def __len__(self) -> int:
        return len(self.tensors['x'])

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.tensors.items()}


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not c.lower().startswith('unnamed')]
    return df[keep].copy()


def _coerce_time_and_sort(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: Optional[str]):
    if time_col is None or time_col not in train_df.columns or time_col not in test_df.columns:
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    train_df = train_df.copy(); test_df = test_df.copy()
    if not pd.api.types.is_numeric_dtype(train_df[time_col]):
        train_df[time_col] = pd.to_datetime(train_df[time_col], errors='coerce')
        test_df[time_col] = pd.to_datetime(test_df[time_col], errors='coerce')
    return train_df.sort_values(time_col).reset_index(drop=True), test_df.sort_values(time_col).reset_index(drop=True)


def _parse_numeric_expected_delta(expected_freq):
    if expected_freq is None:
        return None
    if isinstance(expected_freq, (int, float)):
        return float(expected_freq)
    s = str(expected_freq).strip().lower()
    try:
        return float(s)
    except ValueError:
        pass
    digits = ''.join(ch for ch in s if ch.isdigit() or ch in '.-')
    return float(digits) if digits else None


def _split_contiguous_segments_safe(df: pd.DataFrame, time_col: Optional[str], expected_freq):
    if time_col is None or expected_freq is None or time_col not in df.columns:
        return [df.reset_index(drop=True)]
    series = df[time_col]
    if pd.api.types.is_numeric_dtype(series):
        step = _parse_numeric_expected_delta(expected_freq)
        if step is None:
            return [df.reset_index(drop=True)]
        delta = series.diff().to_numpy()
        bp = np.ones(len(df), dtype=bool)
        if len(df) > 1:
            bp[1:] = ~np.isclose(delta[1:], step, atol=1e-8, rtol=0.0)
        seg_id = np.cumsum(bp)
        return [g.reset_index(drop=True) for _, g in df.groupby(seg_id)]
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors='coerce')
    try:
        expected_delta = pd.Timedelta(expected_freq)
    except Exception:
        return [df.reset_index(drop=True)]
    delta = series.diff()
    bp = delta.ne(expected_delta)
    if len(bp) > 0:
        bp.iloc[0] = True
    seg_id = bp.cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(seg_id)]


def _clean_and_impute_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    train_feat = train_df[feature_cols].copy(); test_feat = test_df[feature_cols].copy()
    for c in feature_cols:
        train_feat[c] = pd.to_numeric(train_feat[c], errors='coerce')
        test_feat[c] = pd.to_numeric(test_feat[c], errors='coerce')
    train_feat = train_feat.replace([np.inf, -np.inf], np.nan)
    test_feat = test_feat.replace([np.inf, -np.inf], np.nan)
    train_feat = train_feat.ffill().bfill(); test_feat = test_feat.ffill().bfill()
    med = train_feat.median()
    train_feat = train_feat.fillna(med); test_feat = test_feat.fillna(med)
    if train_feat.isna().sum().sum() > 0 or test_feat.isna().sum().sum() > 0:
        raise ValueError('NaNs remain after imputation in anomaly dataset.')
    return train_feat, test_feat


def _scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], normalize: str):
    scaler_bundle = build_scaler(normalize)
    train_feat, test_feat = _clean_and_impute_features(train_df, test_df, feature_cols)
    train_scaled = train_df.copy(); test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler_bundle.scaler.fit_transform(train_feat.values)
    test_scaled[feature_cols] = scaler_bundle.scaler.transform(test_feat.values)
    if np.isnan(train_scaled[feature_cols].values).any() or np.isnan(test_scaled[feature_cols].values).any():
        raise ValueError('Scaled anomaly features contain NaNs.')
    return train_scaled, test_scaled, scaler_bundle


def _build_reconstruction_windows(df: pd.DataFrame, feature_cols: list[str], endo_cols: list[str], exo_cols: list[str], seq_len: int, stride: int, time_col: Optional[str], expected_freq):
    segments = _split_contiguous_segments_safe(df, time_col, expected_freq)
    outs = {k: [] for k in ['x', 'y', 'endo_x', 'exo_x', 'x_mark', 'endo_mask', 'exo_mask']}
    segment_lengths = []
    for seg in segments:
        seg = seg.reset_index(drop=True)
        segment_lengths.append(len(seg))
        if len(seg) < seq_len:
            continue
        feat = seg[feature_cols].to_numpy(dtype=np.float32)
        endo = seg[endo_cols].to_numpy(dtype=np.float32)
        exo = seg[exo_cols].to_numpy(dtype=np.float32) if exo_cols else np.zeros((len(seg), 0), dtype=np.float32)
        marks = np.zeros((len(seg), 5), dtype=np.float32)
        for start in range(0, len(seg) - seq_len + 1, stride):
            end = start + seq_len
            outs['x'].append(feat[start:end])
            outs['y'].append(endo[start:end])
            outs['endo_x'].append(endo[start:end])
            outs['exo_x'].append(exo[start:end])
            outs['x_mark'].append(marks[start:end])
            outs['endo_mask'].append(np.ones((seq_len, len(endo_cols)), dtype=np.float32))
            outs['exo_mask'].append(np.ones((seq_len, len(exo_cols)), dtype=np.float32) if exo_cols else np.zeros((seq_len, 0), dtype=np.float32))
    if len(outs['x']) == 0:
        raise ValueError(f'No anomaly windows were created. seq_len={seq_len}, stride={stride}, segments={len(segments)}, lengths={segment_lengths[:10]}')
    return {k: np.stack(v).astype(np.float32) for k, v in outs.items()}, {'num_segments': len(segments), 'segment_lengths': segment_lengths, 'num_windows': len(outs['x'])}


def _extract_pointwise_labels(label_df: pd.DataFrame, time_col: Optional[str]) -> np.ndarray:
    candidate_cols = list(label_df.columns)
    if time_col is not None and time_col in candidate_cols:
        candidate_cols.remove(time_col)
    if not candidate_cols:
        raise ValueError('No usable label column found.')
    label_col = candidate_cols[-1]
    vals = pd.to_numeric(label_df[label_col], errors='coerce').fillna(0).astype(int).to_numpy()
    return (vals > 0).astype(int)


def build_anomaly_loaders(config: dict):
    cfg = config['anomaly']
    data_root = Path(config['paths']['data_root'])
    train_df = _drop_unnamed(pd.read_csv(data_root / cfg['train_file']))
    test_df = _drop_unnamed(pd.read_csv(data_root / cfg['test_file']))
    label_df = _drop_unnamed(pd.read_csv(data_root / cfg['test_label_file']))

    time_col = cfg.get('time_col')
    train_df, test_df = _coerce_time_and_sort(train_df, test_df, time_col)
    if time_col is not None and time_col in label_df.columns:
        if not pd.api.types.is_numeric_dtype(label_df[time_col]):
            label_df[time_col] = pd.to_datetime(label_df[time_col], errors='coerce')
        label_df = label_df.sort_values(time_col).reset_index(drop=True)
    else:
        label_df = label_df.reset_index(drop=True)

    feature_cols = infer_feature_cols(train_df, time_col=time_col)
    endo_cols = cfg.get('endo_cols') or feature_cols
    exo_cols = cfg.get('exo_cols') or []
    exo_cols = [c for c in exo_cols if c in feature_cols and c not in endo_cols]

    train_scaled, test_scaled, scaler_bundle = _scale_features(train_df, test_df, feature_cols, cfg['normalize'])

    seq_len = int(cfg['seq_len']); stride = int(cfg['stride']); expected_freq = cfg.get('expected_freq')
    train_all, train_debug = _build_reconstruction_windows(train_scaled, feature_cols, endo_cols, exo_cols, seq_len, stride, time_col, expected_freq)
    n_total = len(train_all['x'])
    n_val = max(1, int(n_total * float(cfg['val_ratio'])))
    if n_total - n_val < 1:
        n_val = max(1, n_total // 5)
    train_w = {k: v[:-n_val] for k, v in train_all.items()}
    val_w = {k: v[-n_val:] for k, v in train_all.items()}
    test_w, test_debug = _build_reconstruction_windows(test_scaled, feature_cols, endo_cols, exo_cols, seq_len, stride, time_col, expected_freq)

    train_ds = AnomalyWindowDataset(train_w); val_ds = AnomalyWindowDataset(val_w); test_ds = AnomalyWindowDataset(test_w)
    train_cfg = config['train']
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    label_values = _extract_pointwise_labels(label_df, time_col=time_col)
    metadata = {
        'feature_cols': feature_cols,
        'endo_cols': endo_cols,
        'exo_cols': exo_cols,
        'input_dim': len(feature_cols),
        'n_endo': len(endo_cols),
        'n_exo': len(exo_cols),
        'seq_len': seq_len,
        'stride': stride,
        'threshold_mode': cfg.get('threshold_mode', 'val_quantile'),
        'threshold_quantile': cfg.get('threshold_quantile', 0.995),
        'scaler': scaler_bundle,
        'raw_train_df': train_df,
        'raw_test_df': test_df,
        'scaled_train_df': train_scaled,
        'scaled_test_df': test_scaled,
        'test_labels_pointwise': label_values,
        'train_debug': train_debug,
        'test_debug': test_debug,
    }
    print('[Anomaly Loader Build Summary]')
    print(f"  train windows: {len(train_ds)} | val windows: {len(val_ds)} | test windows: {len(test_ds)}")
    print(f"  n_endo={len(endo_cols)} n_exo={len(exo_cols)}")
    return train_loader, val_loader, test_loader, metadata
