"""Time-aware validation split helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def rolling_origin_splits(
    frame: pd.DataFrame,
    date_column: str,
    n_splits: int,
    horizon: int,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield train/validation indices using ordered unique dates."""
    dates = pd.Index(sorted(pd.to_datetime(frame[date_column]).unique()))
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(dates) < n_splits + horizon:
        raise ValueError("Not enough dates for the requested split design.")

    for split_number in range(n_splits, 0, -1):
        validation_end = len(dates) - (split_number - 1) * horizon
        validation_start = validation_end - horizon
        train_dates = dates[:validation_start]
        validation_dates = dates[validation_start:validation_end]
        train_mask = pd.to_datetime(frame[date_column]).isin(train_dates)
        validation_mask = pd.to_datetime(frame[date_column]).isin(validation_dates)
        yield frame.index[train_mask], frame.index[validation_mask]
