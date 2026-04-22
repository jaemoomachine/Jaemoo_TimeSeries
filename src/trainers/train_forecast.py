
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.forecast_dataset import build_forecast_loaders
from src.metrics.forecast_metrics import compute_forecast_metrics
from src.models.model_wrapper import UnifiedTimeSeriesModel
from src.trainers.common import get_device, move_batch_to_device, save_checkpoint
from src.utils.logging import CSVLogger
from src.utils.plotting import save_forecast_plot, save_training_curve


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    preds, gts = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            y = batch['y']
            pred = model(batch)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds.append(pred.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0); gts = np.concatenate(gts, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_forecast_metrics(gts, preds); metrics['loss'] = avg_loss
    return metrics, preds, gts


def run_forecast_experiment(config: dict):
    train_loader, val_loader, test_loader, metadata = build_forecast_loaders(config)
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'transformer')
    model = UnifiedTimeSeriesModel(
        task='forecast',
        input_dim=metadata['input_dim'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg['dim_feedforward'],
        dropout=model_cfg['dropout'],
        pred_len=metadata['pred_len'],
        output_dim=metadata['output_dim'],
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
    logger = CSVLogger(results_root / 'logs' / 'forecast_metrics.csv')
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf'); best_test_preds = best_test_gts = None; best_test_metrics = None
    for epoch in range(1, config['train']['epochs'] + 1):
        train_metrics, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics, _, _ = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history['train_loss'].append(train_metrics['loss']); history['val_loss'].append(val_metrics['loss'])
        row = {'epoch': epoch, 'model_name': model_name, **{f'train_{k}': v for k, v in train_metrics.items()}, **{f'val_{k}': v for k, v in val_metrics.items()}}
        logger.log(row)
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            test_metrics, test_preds, test_gts = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)
            best_test_preds, best_test_gts, best_test_metrics = test_preds, test_gts, test_metrics
            save_checkpoint(model, results_root / 'checkpoints' / 'forecast_best.pt')
    save_training_curve(history, results_root / 'figures' / 'forecast_training_curve.png', title='forecast training curve')
    if best_test_preds is not None:
        save_forecast_plot(best_test_gts[:1], best_test_preds[:1], results_root / 'figures' / 'forecast_example.png', title='forecast example')
    return best_test_metrics
