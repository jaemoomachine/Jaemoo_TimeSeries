from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class ScalerBundle:
    kind: str
    scaler: object

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(x)


def build_scaler(kind: str = 'standard') -> ScalerBundle:
    kind = kind.lower()
    if kind == 'standard':
        return ScalerBundle(kind=kind, scaler=StandardScaler())
    if kind == 'robust':
        return ScalerBundle(kind=kind, scaler=RobustScaler())
    raise ValueError(f'Unsupported scaler kind: {kind}')


def chronological_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:n_train + n_val + n_test].reset_index(drop=True)
    return train, val, test


def fit_transform_by_train(train_df: pd.DataFrame, other_dfs: Sequence[pd.DataFrame], columns: list[str], scaler_kind: str = 'standard'):
    scaler_bundle = build_scaler(scaler_kind)
    train_scaled = train_df.copy()
    train_scaled[columns] = scaler_bundle.scaler.fit_transform(train_df[columns].values)
    scaled_others = []
    for df in other_dfs:
        tmp = df.copy()
        tmp[columns] = scaler_bundle.scaler.transform(df[columns].values)
        scaled_others.append(tmp)
    return train_scaled, scaled_others, scaler_bundle


def split_contiguous_segments(df: pd.DataFrame, time_col: str, expected_delta: str | pd.Timedelta) -> list[pd.DataFrame]:
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    expected_delta = pd.Timedelta(expected_delta)
    delta = df[time_col].diff()
    breakpoints = delta.ne(expected_delta)
    breakpoints.iloc[0] = True
    segment_ids = breakpoints.cumsum()
    segments = [g.reset_index(drop=True) for _, g in df.groupby(segment_ids)]
    return segments


def infer_feature_cols(df: pd.DataFrame, time_col: Optional[str], exclude_cols: Optional[Sequence[str]] = None) -> list[str]:
    exclude = set(exclude_cols or [])
    if time_col is not None:
        exclude.add(time_col)
    return [c for c in df.columns if c not in exclude]
