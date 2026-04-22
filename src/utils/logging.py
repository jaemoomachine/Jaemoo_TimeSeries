from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .io import ensure_dir


class CSVLogger:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        ensure_dir(self.file_path.parent)
        self.header_written = self.file_path.exists() and self.file_path.stat().st_size > 0

    def log(self, row: Dict[str, float | int | str]) -> None:
        with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)
