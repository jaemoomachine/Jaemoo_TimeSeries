
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    out_dir = csv_path.parent

    if args.task == 'forecast':
        metric = 'rmse'
    elif args.task == 'anomaly':
        metric = 'pa0_f1'
    else:
        metric = 'accuracy'

    df = df.sort_values(metric, ascending=(args.task == 'forecast'))
    plt.figure(figsize=(8, 4))
    plt.bar(df['model_name'], df[metric])
    plt.title(f'{args.task} comparison ({metric})')
    plt.tight_layout()
    plt.savefig(out_dir / f'{args.task}_comparison.png')
    plt.close()
    print(df)


if __name__ == '__main__':
    main()
