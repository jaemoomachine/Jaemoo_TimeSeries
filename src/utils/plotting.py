from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_training_curve(history: Dict[str, list], save_path: str | Path, title: str = 'training_curve') -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_forecast_plot(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path, title: str = 'forecast') -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.reshape(-1), label='true')
    plt.plot(y_pred.reshape(-1), label='pred')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_anomaly_plot(score: np.ndarray, threshold: float, labels: Optional[np.ndarray], save_path: str | Path, title: str = 'anomaly_score') -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(score, label='score')
    plt.axhline(threshold, linestyle='--', label='threshold')
    if labels is not None:
        anomaly_idx = np.where(labels == 1)[0]
        if len(anomaly_idx) > 0:
            plt.scatter(anomaly_idx, score[anomaly_idx], s=10, label='gt_anomaly')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], save_path: str | Path, title: str = 'confusion_matrix') -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
