"""Preprocessing utilities for building model-ready tables."""

from __future__ import annotations

import pandas as pd

MERGE_KEYS = ("period_id", "jurisdiction")
TARGET_COLUMN = "rate_per_10000_ed_visits"


def build_training_table(covariates: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Merge target rows with covariates without dropping training labels."""
    missing_covariate_keys = [key for key in MERGE_KEYS if key not in covariates.columns]
    missing_target_columns = [
        column for column in (*MERGE_KEYS, TARGET_COLUMN) if column not in targets.columns
    ]

    if missing_covariate_keys:
        raise ValueError(f"Covariates are missing required merge key(s): {missing_covariate_keys}")
    if missing_target_columns:
        raise ValueError(f"Targets are missing required column(s): {missing_target_columns}")

    merged = targets.merge(
        covariates,
        on=list(MERGE_KEYS),
        how="left",
        validate="many_to_one",
    )

    if len(merged) != len(targets):
        raise ValueError(
            f"Training label count changed during merge: before={len(targets)}, after={len(merged)}"
        )
    if TARGET_COLUMN not in merged.columns:
        raise ValueError(f"Target column is missing after merge: {TARGET_COLUMN}")

    return merged
