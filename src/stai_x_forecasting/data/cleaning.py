"""Data validation and cleaning helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

TARGET_COLUMN = "rate_per_10000_ed_visits"
GROUP_COLUMN = "jurisdiction"
TEXT_COLUMN = "state_doh_release"


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    df_name: str,
) -> None:
    """Raise a clear error when required columns are missing."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{df_name} is missing required column(s): {missing_columns}")


def check_duplicates(df: pd.DataFrame, key_columns: Sequence[str]) -> int:
    """Return duplicate row count based on key columns."""
    validate_required_columns(df, key_columns, "dataframe")
    return int(df.duplicated(subset=list(key_columns)).sum())


def clean_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean covariates without dropping rows."""
    cleaned = df.copy()
    validate_required_columns(cleaned, [GROUP_COLUMN], "covariates")

    original_release = (
        cleaned[TEXT_COLUMN]
        if TEXT_COLUMN in cleaned.columns
        else pd.Series(pd.NA, index=cleaned.index, dtype="object")
    )
    cleaned["has_state_doh_release"] = original_release.notna().astype("int8")

    if TEXT_COLUMN in cleaned.columns:
        cleaned[TEXT_COLUMN] = cleaned[TEXT_COLUMN].fillna("No release")
    else:
        cleaned[TEXT_COLUMN] = "No release"

    cleaned["state_doh_release_length"] = cleaned[TEXT_COLUMN].astype(str).str.len()

    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()
    for column in numeric_columns:
        if not cleaned[column].isna().any():
            continue
        jurisdiction_medians = cleaned.groupby(GROUP_COLUMN)[column].transform("median")
        cleaned[column] = cleaned[column].fillna(jurisdiction_medians)
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned


def clean_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the merged training table without dropping rows."""
    validate_required_columns(df, [TARGET_COLUMN], "training table")
    if df[TARGET_COLUMN].isna().any():
        missing_count = int(df[TARGET_COLUMN].isna().sum())
        raise ValueError(f"Training target has missing values: {missing_count}")

    return clean_covariates(df)
