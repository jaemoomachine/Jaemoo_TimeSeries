from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'mse': mse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
    }
