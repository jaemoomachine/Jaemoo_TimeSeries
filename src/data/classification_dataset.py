
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseTimeSeriesDataset
from .tsfile_parser import load_equal_length_multivariate_ts


class ClassificationDataset(BaseTimeSeriesDataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(task='classification')
        x_blc = np.transpose(x, (0, 2, 1)).astype(np.float32)
        N, L, C = x_blc.shape
        self.data = {
            'x': torch.tensor(x_blc, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.long),
            'endo_x': torch.tensor(x_blc, dtype=torch.float32),
            'exo_x': torch.zeros((N, L, 0), dtype=torch.float32),
            'x_mark': torch.zeros((N, L, 5), dtype=torch.float32),
            'endo_mask': torch.ones((N, L, C), dtype=torch.float32),
            'exo_mask': torch.zeros((N, L, 0), dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.data['x'])

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.data.items()}


def _fit_channel_standardizer(x_train: np.ndarray):
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def build_classification_loaders(config: dict):
    cfg = config['classification']
    data_root = Path(config['paths']['data_root'])
    x_train, y_train, class_names = load_equal_length_multivariate_ts(data_root / cfg['train_file'])
    x_test, y_test, class_names_test = load_equal_length_multivariate_ts(data_root / cfg['test_file'])

    if class_names != class_names_test:
        raise ValueError('Train/test class name order mismatch.')

    if cfg.get('normalize', 'standard') == 'standard':
        mean, std = _fit_channel_standardizer(x_train)
        x_train = _apply_standardize(x_train, mean, std)
        x_test = _apply_standardize(x_test, mean, std)
    else:
        mean = std = None

    train_ds = ClassificationDataset(x_train, y_train)
    test_ds = ClassificationDataset(x_test, y_test)

    train_cfg = config['train']
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    metadata = {
        'input_dim': x_train.shape[1],
        'seq_len': x_train.shape[2],
        'num_classes': len(class_names),
        'class_names': class_names,
        'normalizer_mean': mean,
        'normalizer_std': std,
        'n_endo': x_train.shape[1],
        'n_exo': 0,
        'endo_cols': [f'channel_{i}' for i in range(x_train.shape[1])],
        'exo_cols': [],
    }
    return train_loader, test_loader, metadata
