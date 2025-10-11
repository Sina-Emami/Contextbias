"""Shared file I/O helpers for the data processing pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Sequence


def write_atomic_text(path: Path, data: str, *, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding=encoding, dir=path.parent, delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)


def write_atomic_csv(
    path: Path,
    rows: Iterable[Sequence[str]],
    *,
    encoding: str = "utf-8",
) -> None:
    """Write CSV rows to ``path`` atomically, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding=encoding, newline="", dir=path.parent, delete=False) as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerows(rows)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)

