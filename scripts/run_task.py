from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.trainers.train_anomaly import run_anomaly_experiment
from src.trainers.train_classification import run_classification_experiment
from src.trainers.train_forecast import run_forecast_experiment
from src.utils.io import load_config, resolve_config_paths, save_json
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['forecast', 'anomaly', 'classification'])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), ROOT)
    set_seed(config.get('seed', 42))

    if args.task == 'forecast':
        metrics = run_forecast_experiment(config)
    elif args.task == 'anomaly':
        metrics = run_anomaly_experiment(config)
    elif args.task == 'classification':
        metrics = run_classification_experiment(config)
    else:
        raise ValueError(args.task)

    results_root = Path(config['paths']['results_root'])
    save_json(metrics, results_root / 'logs' / f'{args.task}_best_metrics.json')
    print(metrics)


if __name__ == '__main__':
    main()
