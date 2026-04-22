from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def _parse_bool_token(line: str) -> bool:
    token = line.strip().split()[-1].lower()
    if token == 'true':
        return True
    if token == 'false':
        return False
    raise ValueError(f'Invalid boolean token: {token}')


def load_equal_length_multivariate_ts(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = Path(path)
    metadata = {
        'timestamps': False,
        'univariate': False,
        'classlabel': True,
        'class_values': [],
        'data_started': False,
    }
    rows: list[list[np.ndarray]] = []
    labels: list[str] = []

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            low = line.lower()
            if low.startswith('@timestamps'):
                metadata['timestamps'] = _parse_bool_token(line)
                if metadata['timestamps']:
                    raise ValueError('This parser does not support timestamped .ts files.')
                continue
            if low.startswith('@univariate'):
                metadata['univariate'] = _parse_bool_token(line)
                continue
            if low.startswith('@classlabel'):
                parts = line.split()
                metadata['classlabel'] = parts[1].lower() == 'true'
                metadata['class_values'] = parts[2:] if len(parts) > 2 else []
                continue
            if low.startswith('@data'):
                metadata['data_started'] = True
                continue
            if line.startswith('@'):
                continue
            if not metadata['data_started']:
                continue

            parts = line.split(':')
            if metadata['classlabel']:
                *dim_parts, label = parts
                labels.append(label)
            else:
                dim_parts = parts

            dims = []
            for dim in dim_parts:
                dim = dim.strip()
                if dim == '':
                    raise ValueError('Empty dimension found in .ts row.')
                values = [np.nan if v == '?' else float(v) for v in dim.split(',') if v != '']
                dims.append(np.array(values, dtype=np.float32))
            rows.append(dims)

    if not rows:
        raise ValueError(f'No rows were parsed from {path}')

    n_dims = len(rows[0])
    lengths = {len(dim) for row in rows for dim in row}
    if len(lengths) != 1:
        raise ValueError('This parser currently supports equal-length series only.')

    series_len = list(lengths)[0]
    n_cases = len(rows)
    x = np.zeros((n_cases, n_dims, series_len), dtype=np.float32)
    for i, row in enumerate(rows):
        if len(row) != n_dims:
            raise ValueError('Inconsistent number of dimensions in .ts file.')
        for d, arr in enumerate(row):
            x[i, d, :] = arr

    class_values = metadata['class_values']
    y_raw = np.array(labels)
    if class_values:
        class_to_idx = {c: i for i, c in enumerate(class_values)}
        y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)
    else:
        uniques = sorted(set(labels))
        class_to_idx = {c: i for i, c in enumerate(uniques)}
        y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)
        class_values = uniques
    return x, y, class_values
