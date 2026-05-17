"""Simple baseline forecasters for validation sanity checks."""

from __future__ import annotations

import pandas as pd


class LastValueForecaster:
    """Predict the last observed target value per group."""

    def __init__(self, group_column: str, target_column: str) -> None:
        self.group_column = group_column
        self.target_column = target_column
        self.global_value_: float | None = None
        self.group_values_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "LastValueForecaster":
        self.global_value_ = float(frame[self.target_column].mean())
        self.group_values_ = frame.groupby(self.group_column)[self.target_column].last()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.group_values_ is None or self.global_value_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        predictions = frame[self.group_column].map(self.group_values_)
        return predictions.fillna(self.global_value_)


class GroupMeanForecaster:
    """Predict the historical mean target value per group."""

    def __init__(self, group_column: str, target_column: str) -> None:
        self.group_column = group_column
        self.target_column = target_column
        self.global_value_: float | None = None
        self.group_values_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "GroupMeanForecaster":
        self.global_value_ = float(frame[self.target_column].mean())
        self.group_values_ = frame.groupby(self.group_column)[self.target_column].mean()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.group_values_ is None or self.global_value_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        predictions = frame[self.group_column].map(self.group_values_)
        return predictions.fillna(self.global_value_)
