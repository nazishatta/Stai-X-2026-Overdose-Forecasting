"""Historical target aggregate features."""

from __future__ import annotations

from typing import Any

import pandas as pd

TARGET_COLUMN = "rate_per_10000_ed_visits"


def build_target_history_features(reference_targets: pd.DataFrame) -> dict[str, Any]:
    """Compute historical aggregate features from reference targets only."""
    required_columns = ["jurisdiction", "overdose_category", TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in reference_targets.columns]
    if missing_columns:
        raise ValueError(f"reference_targets is missing required column(s): {missing_columns}")

    targets = reference_targets.copy()
    return {
        "jurisdiction_category": (
            targets.groupby(["jurisdiction", "overdose_category"], as_index=False)[TARGET_COLUMN]
            .mean()
            .rename(columns={TARGET_COLUMN: "jurisdiction_category_mean_rate"})
        ),
        "category": (
            targets.groupby("overdose_category", as_index=False)[TARGET_COLUMN]
            .mean()
            .rename(columns={TARGET_COLUMN: "category_mean_rate"})
        ),
        "jurisdiction": (
            targets.groupby("jurisdiction", as_index=False)[TARGET_COLUMN]
            .mean()
            .rename(columns={TARGET_COLUMN: "jurisdiction_mean_rate"})
        ),
        "global": float(targets[TARGET_COLUMN].mean()),
    }


def add_target_history_features(
    df: pd.DataFrame,
    reference_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Join historical target features onto a copied dataframe without dropping rows."""
    features = build_target_history_features(reference_targets)
    output = df.copy()

    output = output.merge(
        features["jurisdiction_category"],
        on=["jurisdiction", "overdose_category"],
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        features["category"],
        on="overdose_category",
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        features["jurisdiction"],
        on="jurisdiction",
        how="left",
        validate="many_to_one",
    )

    global_mean = features["global"]
    output["global_mean_rate"] = global_mean
    output["category_mean_rate"] = output["category_mean_rate"].fillna(global_mean)
    output["jurisdiction_category_mean_rate"] = output[
        "jurisdiction_category_mean_rate"
    ].fillna(output["category_mean_rate"])
    output["jurisdiction_category_mean_rate"] = output[
        "jurisdiction_category_mean_rate"
    ].fillna(global_mean)
    output["jurisdiction_mean_rate"] = output["jurisdiction_mean_rate"].fillna(global_mean)

    return output
