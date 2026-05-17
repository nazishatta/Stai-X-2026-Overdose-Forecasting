"""Temporal feature helpers."""

from __future__ import annotations

import pandas as pd


def add_calendar_features(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add basic calendar features from a weekly date column."""
    output = frame.copy()
    dates = pd.to_datetime(output[date_column])
    output["year"] = dates.dt.year
    output["month"] = dates.dt.month
    output["weekofyear"] = dates.dt.isocalendar().week.astype("int16")
    return output
