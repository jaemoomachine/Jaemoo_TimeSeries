from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_config_paths(config: Dict[str, Any], base_dir: str | Path) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    paths = config.get("paths", {})
    resolved = dict(config)
    resolved_paths = dict(paths)
    for key in ["data_root", "results_root"]:
        if key in resolved_paths:
            raw = Path(resolved_paths[key])
            resolved_paths[key] = str(raw if raw.is_absolute() else (base_dir / raw))
    resolved["paths"] = resolved_paths
    return resolved
