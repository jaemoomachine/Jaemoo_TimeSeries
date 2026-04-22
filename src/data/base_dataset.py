from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import Dataset


class BaseTimeSeriesDataset(Dataset):
    def __init__(self, task: str):
        self.task = task

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError
