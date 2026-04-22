
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.trainers.train_anomaly import run_anomaly_experiment
from src.trainers.train_classification import run_classification_experiment
from src.trainers.train_forecast import run_forecast_experiment
from src.utils.io import load_config, resolve_config_paths, save_json
from src.utils.seed import set_seed


RUNNERS = {
    'forecast': run_forecast_experiment,
    'anomaly': run_anomaly_experiment,
    'classification': run_classification_experiment,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=list(RUNNERS.keys()))
    parser.add_argument('--configs', nargs='+', required=True)
    args = parser.parse_args()

    rows = []
    for config_path in args.configs:
        cfg = resolve_config_paths(load_config(config_path), ROOT)
        set_seed(cfg.get('seed', 42))
        metrics = RUNNERS[args.task](cfg)
        row = {'config': config_path, 'task': args.task, 'model_name': cfg['model'].get('name', 'transformer'), **metrics}
        rows.append(row)
        save_json(row, Path(cfg['paths']['results_root']) / 'logs' / f'{args.task}_summary.json')
        print(row)

    if rows:
        out_dir = ROOT / 'results' / 'benchmarks'
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f'{args.task}_benchmark_summary.csv', index=False, encoding='utf-8-sig')
        print(df)


if __name__ == '__main__':
    main()
