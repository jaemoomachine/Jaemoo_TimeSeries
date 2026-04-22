from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from .preprocess import infer_feature_cols


def _save_series_plot(series: pd.Series, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(series.values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _save_hist_plot(series: pd.Series, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna().values, bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _save_corr_plot(df: pd.DataFrame, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect='auto')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_tabular_time_series_analysis(df: pd.DataFrame, save_dir: str | Path, time_col: Optional[str] = None, max_features_for_plots: int = 10) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if time_col is not None and time_col in df.columns:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    feature_cols = infer_feature_cols(df, time_col=time_col)
    stats_df = df[feature_cols].describe().T
    stats_df['skew'] = [skew(df[c].dropna().values) if df[c].dropna().shape[0] > 2 else np.nan for c in feature_cols]
    stats_df.to_csv(save_dir / 'descriptive_statistics.csv', encoding='utf-8-sig')

    missing_df = df[feature_cols].isna().sum().to_frame(name='missing_count')
    missing_df['missing_ratio'] = missing_df['missing_count'] / len(df)
    missing_df.to_csv(save_dir / 'missing_values.csv', encoding='utf-8-sig')

    adf_rows = []
    for col in feature_cols[:max_features_for_plots]:
        series = df[col].dropna().values

        if len(series) > 20:
            try:
                stat, pval, usedlag, nobs, critical_values, icbest = adfuller(series)
            except Exception:
                stat, pval, usedlag, nobs, icbest = np.nan, np.nan, np.nan, np.nan, np.nan
                critical_values = {}
        else:
            stat, pval, usedlag, nobs, icbest = np.nan, np.nan, np.nan, np.nan, np.nan
            critical_values = {}

        adf_rows.append({
            "feature": col,
            "adf_stat": stat,
            "p_value": pval,
      #      "used_lag": usedlag,
            "n_obs": nobs,
      #      "icbest": icbest,
            "critical_value_1%": critical_values.get("1%", np.nan),
            "critical_value_5%": critical_values.get("5%", np.nan),
            "critical_value_10%": critical_values.get("10%", np.nan),
        })

    pd.DataFrame(adf_rows).to_csv(
        save_dir / "adf_results.csv",
        index=False,
        encoding="utf-8-sig"
    )
    _save_corr_plot(df[feature_cols], save_dir / 'correlation.png', 'correlation heatmap')

    for col in feature_cols[:max_features_for_plots]:
        _save_series_plot(df[col], save_dir / f'{col}_series.png', f'{col} series')
        _save_hist_plot(df[col], save_dir / f'{col}_hist.png', f'{col} histogram')

        if len(df[col].dropna()) >= 48:
            rolling = df[col].rolling(window=24)
            plt.figure(figsize=(10, 4))
            plt.plot(df[col].values, label='series')
            plt.plot(rolling.mean().values, label='rolling_mean')
            plt.plot(rolling.std().values, label='rolling_std')
            plt.title(f'{col} rolling mean/std')
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f'{col}_rolling.png')
            plt.close()

    if feature_cols:
        col = feature_cols[0]
        series = df[col].dropna()
        if len(series) >= 2 * 24:
            try:
                result = seasonal_decompose(series.iloc[: min(len(series), 24 * 14)], period=24, model='additive')
                fig = result.plot()
                fig.set_size_inches(10, 8)
                fig.tight_layout()
                fig.savefig(save_dir / f'{col}_decomposition.png')
                plt.close(fig)
            except Exception:
                pass

    return {
        'n_rows': len(df),
        'n_features': len(feature_cols),
        'features': feature_cols,
    }

def run_classification_analysis(x: np.ndarray, y: np.ndarray, class_names: list[str], save_dir: str | Path) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # x: [N, C, L]
    N, C, L = x.shape

    shape_df = pd.DataFrame([{
        "num_samples": int(N),
        "num_channels": int(C),
        "seq_len": int(L),
        "num_classes": int(len(class_names)),
        "total_missing": int(np.isnan(x).sum()),
    }])
    shape_df.to_csv(save_dir / "shape_summary.csv", index=False, encoding="utf-8-sig")

    counts = np.bincount(y, minlength=len(class_names))
    class_dist_df = pd.DataFrame({
        "class": class_names,
        "count": counts
    })
    class_dist_df.to_csv(save_dir / "class_distribution.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts)
    plt.title("class distribution")
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png")
    plt.close()

    stat_rows = []
    missing_rows = []
    adf_rows = []

    for c in range(C):
        vals = x[:, c, :].reshape(-1)
        s = pd.Series(vals)

        stat_rows.append({
            "channel": f"channel_{c}",
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "25%": float(s.quantile(0.25)),
            "50%": float(s.quantile(0.50)),
            "75%": float(s.quantile(0.75)),
            "max": float(s.max()),
            "skew": float(s.skew()),
        })

        missing_rows.append({
            "channel": f"channel_{c}",
            "missing_count": int(s.isna().sum()),
            "missing_ratio": float(s.isna().mean()),
        })


        seq = s.dropna().values
        if len(seq) > 20:
            try:
                stat, pval, usedlag, nobs, critical_values, icbest = adfuller(seq)
            except Exception:
                stat, pval, usedlag, nobs, icbest = np.nan, np.nan, np.nan, np.nan, np.nan
                critical_values = {}
        else:
            stat, pval, usedlag, nobs, icbest = np.nan, np.nan, np.nan, np.nan, np.nan
            critical_values = {}

        cv_1 = critical_values.get("1%", np.nan)
        cv_5 = critical_values.get("5%", np.nan)
        cv_10 = critical_values.get("10%", np.nan)

        adf_rows.append({
            "channel": f"channel_{c}",
            "adf_stat": stat,
            "p_value": pval,
            "used_lag": usedlag,
            "n_obs": nobs,
            "icbest": icbest,
            "critical_value_1%": cv_1,
            "critical_value_5%": cv_5,
            "critical_value_10%": cv_10,
            "stationary_at_1%": bool(stat < cv_1) if pd.notna(stat) and pd.notna(cv_1) else np.nan,
            "stationary_at_5%": bool(stat < cv_5) if pd.notna(stat) and pd.notna(cv_5) else np.nan,
            "stationary_at_10%": bool(stat < cv_10) if pd.notna(stat) and pd.notna(cv_10) else np.nan,
        })

    pd.DataFrame(stat_rows).set_index("channel").to_csv(
        save_dir / "descriptive_statistics.csv",
        encoding="utf-8-sig"
    )
    pd.DataFrame(missing_rows).to_csv(
        save_dir / "missing_values.csv",
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(adf_rows).to_csv(
        save_dir / "adf_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for c in range(min(C, 4)):
        plt.figure(figsize=(10, 4))
        for k in range(len(class_names)):
            cls_mean = x[y == k, c, :].mean(axis=0)
            plt.plot(cls_mean, label=class_names[k])
        plt.title(f"channel_{c}_class_mean_pattern")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"channel_{c}_class_mean.png")
        plt.close()

    summary = {
        "n_samples": int(N),
        "n_channels": int(C),
        "seq_len": int(L),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
    }
    return summary