"""Local validation helpers for period-based holdout scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_period_order_from_targets(targets: pd.DataFrame) -> list[str]:
    """Return period_id values in first-seen order from target rows."""
    if "period_id" not in targets.columns:
        raise ValueError("targets is missing required column: period_id")
    return targets["period_id"].drop_duplicates().tolist()


def make_holdout_split(
    targets: pd.DataFrame,
    n_holdout_periods: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split targets using the last periods from target file order as validation."""
    if n_holdout_periods <= 0:
        raise ValueError("n_holdout_periods must be positive.")

    period_order = get_period_order_from_targets(targets)
    if len(period_order) <= n_holdout_periods:
        raise ValueError(
            "Not enough periods for holdout split: "
            f"periods={len(period_order)}, n_holdout_periods={n_holdout_periods}"
        )

    train_periods = period_order[:-n_holdout_periods]
    valid_periods = period_order[-n_holdout_periods:]

    train_targets = targets[targets["period_id"].isin(train_periods)].copy()
    valid_targets = targets[targets["period_id"].isin(valid_periods)].copy()

    return train_targets, valid_targets, train_periods, valid_periods


def rmse(y_true, y_pred) -> float:
    """Compute root mean squared error."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2)))
