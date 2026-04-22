
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.analysis import run_classification_analysis, run_tabular_time_series_analysis
from src.data.anomaly_dataset import build_anomaly_loaders
from src.data.classification_dataset import _apply_standardize, _fit_channel_standardizer
from src.data.forecast_dataset import build_forecast_loaders
from src.data.tsfile_parser import load_equal_length_multivariate_ts
from src.utils.io import load_config, resolve_config_paths, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['forecast', 'anomaly', 'classification'])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), ROOT)
    data_root = Path(config['paths']['data_root'])
    results_root = Path(config['paths']['results_root']) / 'analysis' / args.task

    if args.task == 'forecast':
        cfg = config['forecast']
        df = pd.read_csv(data_root / cfg['file_name'])
        raw_summary = run_tabular_time_series_analysis(df, results_root / 'raw', time_col=cfg.get('time_col'))
        _, _, _, metadata = build_forecast_loaders(config)
        processed_summary = run_tabular_time_series_analysis(metadata['scaled_splits']['train'], results_root / 'processed_train', time_col=cfg.get('time_col'))
    elif args.task == 'anomaly':
        cfg = config['anomaly']
        train_df = pd.read_csv(data_root / cfg['train_file'])
        raw_summary = run_tabular_time_series_analysis(train_df, results_root / 'raw', time_col=cfg.get('time_col'))
        _, _, _, metadata = build_anomaly_loaders(config)
        processed_summary = run_tabular_time_series_analysis(metadata['scaled_train_df'], results_root / 'processed_train', time_col=cfg.get('time_col'))
    elif args.task == 'classification':
        cfg = config['classification']
        x_train, y_train, class_names = load_equal_length_multivariate_ts(data_root / cfg['train_file'])
        raw_summary = run_classification_analysis(x_train, y_train, class_names, results_root / 'raw')
        if cfg.get('normalize', 'standard') == 'standard':
            mean, std = _fit_channel_standardizer(x_train)
            x_train = _apply_standardize(x_train, mean, std)
        processed_summary = run_classification_analysis(x_train, y_train, class_names, results_root / 'processed_train')
    else:
        raise ValueError(args.task)

    summary = {'raw': raw_summary, 'processed_train': processed_summary}
    save_json(summary, results_root / 'summary.json')
    print(summary)


if __name__ == '__main__':
    main()
