
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.anomaly_dataset import build_anomaly_loaders
from src.metrics.anomaly_metrics import compute_anomaly_metrics, reconstruction_scores, select_threshold_from_val
from src.models.model_wrapper import UnifiedTimeSeriesModel
from src.trainers.common import get_device, move_batch_to_device, save_checkpoint
from src.utils.logging import CSVLogger
from src.utils.plotting import save_anomaly_plot, save_training_curve


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    xs, preds = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            y = batch['y']
            pred = model(batch)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError('Loss became NaN/Inf during anomaly training.')
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * y.size(0)
            xs.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    xs = np.concatenate(xs, axis=0); preds = np.concatenate(preds, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, xs, preds


def _aggregate_window_scores(scores_2d: np.ndarray, seq_len: int, stride: int, total_length: int | None = None) -> np.ndarray:
    n_windows = scores_2d.shape[0]
    if total_length is None:
        total_length = (n_windows - 1) * stride + seq_len
    sums = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)
    for i in range(n_windows):
        start = i * stride; end = min(start + seq_len, total_length)
        local = scores_2d[i, : end - start]
        sums[start:end] += local; counts[start:end] += 1
    counts = np.where(counts == 0, 1.0, counts)
    return sums / counts


def run_anomaly_experiment(config: dict):
    train_loader, val_loader, test_loader, metadata = build_anomaly_loaders(config)
    model_cfg = config['model']; model_name = model_cfg.get('name', 'transformer')
    model = UnifiedTimeSeriesModel(
        task='anomaly',
        input_dim=metadata['input_dim'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg['dim_feedforward'],
        dropout=model_cfg['dropout'],
        output_dim=metadata['n_endo'],
        pooling=model_cfg.get('pooling', 'mean'),
        model_name=model_name,
        seq_len=metadata['seq_len'],
        n_endo=metadata['n_endo'],
        n_exo=metadata['n_exo'],
        patch_len=model_cfg.get('patch_len', 16),
        patch_stride=model_cfg.get('patch_stride', 16),
        patch_padding=model_cfg.get('patch_padding', 0),
        n_globaltokens=model_cfg.get('n_globaltokens', 1),
        attn_dropout=model_cfg.get('attn_dropout', 0.0),
        revin_affine=model_cfg.get('revin_affine', True),
        revin_subtract_last=model_cfg.get('revin_subtract_last', False),
        tau_hidden_dim=model_cfg.get('tau_hidden_dim', 64),
        tau_max=model_cfg.get('tau_max', 20.0),
        delta_clip=model_cfg.get('delta_clip', 10.0),
    )
    device = get_device(config['train']['device']); model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'], foreach=False)
    results_root = Path(config['paths']['results_root'])
    logger = CSVLogger(results_root / 'logs' / 'anomaly_metrics.csv')
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf'); best_metrics = None
    for epoch in range(1, config['train']['epochs'] + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_x, val_pred = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        logger.log({'epoch': epoch, 'model_name': model_name, 'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, results_root / 'checkpoints' / 'anomaly_best.pt')
            val_scores = reconstruction_scores(val_x, val_pred)
            val_point_scores = _aggregate_window_scores(val_scores, seq_len=metadata['seq_len'], stride=metadata['stride'])
            threshold = select_threshold_from_val(val_point_scores, mode=metadata['threshold_mode'], quantile=metadata['threshold_quantile'])
            test_loss, test_x, test_pred = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)
            test_scores = reconstruction_scores(test_x, test_pred)
            point_labels = metadata['test_labels_pointwise']
            point_scores = _aggregate_window_scores(test_scores, seq_len=metadata['seq_len'], stride=metadata['stride'], total_length=len(point_labels))
            best_metrics = compute_anomaly_metrics(point_scores, point_labels, threshold)
            best_metrics['test_loss'] = test_loss
            save_anomaly_plot(point_scores, threshold, point_labels, results_root / 'figures' / 'anomaly_scores.png', title='anomaly score and threshold')
    save_training_curve(history, results_root / 'figures' / 'anomaly_training_curve.png', title='anomaly training curve')
    if best_metrics is None:
        raise RuntimeError('best_metrics is None. Validation never improved, likely due to NaN/Inf losses.')
    return best_metrics
