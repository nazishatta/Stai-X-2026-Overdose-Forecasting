"""Input/output helpers for competition tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet table based on file suffix."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")
