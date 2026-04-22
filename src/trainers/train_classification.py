
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.classification_dataset import build_classification_loaders
from src.metrics.classification_metrics import compute_classification_metrics
from src.models.model_wrapper import UnifiedTimeSeriesModel
from src.trainers.common import get_device, move_batch_to_device, save_checkpoint
from src.utils.logging import CSVLogger
from src.utils.plotting import save_confusion_matrix, save_training_curve


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    logits_list, labels_list = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            y = batch['y']
            logits = model(batch)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * y.size(0)
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())
    logits_all = np.concatenate(logits_list, axis=0); labels_all = np.concatenate(labels_list, axis=0)
    preds = logits_all.argmax(axis=1)
    metrics = compute_classification_metrics(labels_all, preds); metrics['loss'] = total_loss / len(loader.dataset)
    return metrics, labels_all, preds


def run_classification_experiment(config: dict):
    train_loader, test_loader, metadata = build_classification_loaders(config)
    model_cfg = config['model']; model_name = model_cfg.get('name', 'transformer')
    model = UnifiedTimeSeriesModel(
        task='classification',
        input_dim=metadata['input_dim'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg['dim_feedforward'],
        dropout=model_cfg['dropout'],
        num_classes=metadata['num_classes'],
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
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'], foreach=False)
    results_root = Path(config['paths']['results_root'])
    logger = CSVLogger(results_root / 'logs' / 'classification_metrics.csv')
    history = {'train_loss': [], 'train_acc': []}
    best_acc = -1.0; best_metrics = None; best_preds = best_labels = None
    for epoch in range(1, config['train']['epochs'] + 1):
        train_metrics, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_metrics, labels, preds = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        history['train_loss'].append(train_metrics['loss']); history['train_acc'].append(train_metrics['accuracy'])
        logger.log({'epoch': epoch, 'model_name': model_name, **{f'train_{k}': v for k, v in train_metrics.items()}, **{f'test_{k}': v for k, v in test_metrics.items()}})
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']; best_metrics = test_metrics; best_preds = preds; best_labels = labels
            save_checkpoint(model, results_root / 'checkpoints' / 'classification_best.pt')
    save_training_curve(history, results_root / 'figures' / 'classification_training_curve.png', title='classification training curve')
    if best_preds is not None:
        save_confusion_matrix(best_labels, best_preds, metadata['class_names'], results_root / 'figures' / 'classification_confusion_matrix.png')
    return best_metrics
