from __future__ import annotations

import numpy as np


def reconstruction_scores(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    err = np.abs(x_true - x_pred)
    return err.mean(axis=-1)


def precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(int)
    gt = gt.astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def find_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = labels.astype(int)
    segments = []
    start = None
    for i, v in enumerate(labels):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def apply_pa_k(pred: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    adjusted = pred.astype(int).copy()
    segments = find_segments(gt)
    for s, e in segments:
        seg = adjusted[s:e + 1]
        ratio = 100.0 * seg.sum() / max(1, len(seg))
        if ratio > k:
            adjusted[s:e + 1] = 1
    return adjusted


def select_threshold_from_val(val_scores: np.ndarray, mode: str = 'quantile', quantile: float = 0.995) -> float:
    if mode == 'quantile' or mode == 'val_quantile':
        return float(np.quantile(val_scores, quantile))
    raise ValueError(f'Unsupported threshold selection mode: {mode}')


def compute_anomaly_metrics(point_scores: np.ndarray, point_labels: np.ndarray, threshold: float) -> dict:
    pred = (point_scores > threshold).astype(int)
    raw = precision_recall_f1(pred, point_labels)
    pa0 = precision_recall_f1(apply_pa_k(pred, point_labels, 0), point_labels)
    pa100 = precision_recall_f1(apply_pa_k(pred, point_labels, 100), point_labels)
    return {
        'threshold': float(threshold),
        'raw_precision': raw['precision'],
        'raw_recall': raw['recall'],
        'raw_f1': raw['f1'],
        'pa0_precision': pa0['precision'],
        'pa0_recall': pa0['recall'],
        'pa0_f1': pa0['f1'],
        'pa100_precision': pa100['precision'],
        'pa100_recall': pa100['recall'],
        'pa100_f1': pa100['f1'],
    }
