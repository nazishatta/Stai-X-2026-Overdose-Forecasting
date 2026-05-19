"""Historical target aggregate features."""

from __future__ import annotations

from typing import Any

import pandas as pd

TARGET_COLUMN = "rate_per_10000_ed_visits"


def build_target_history_features(reference_targets: pd.DataFrame) -> dict[str, Any]:
    """Compute historical aggregate features from reference targets only."""
    required_columns = ["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in reference_targets.columns]
    if missing_columns:
        raise ValueError(f"reference_targets is missing required column(s): {missing_columns}")

    targets = reference_targets.copy()
    period_order = targets["period_id"].drop_duplicates().tolist()

    def recent_mean(period_count: int) -> pd.DataFrame:
        recent_periods = period_order[-period_count:]
        return (
            targets[targets["period_id"].isin(recent_periods)]
            .groupby(["jurisdiction", "overdose_category"], as_index=False)[TARGET_COLUMN]
            .mean()
            .rename(columns={TARGET_COLUMN: f"recent_{period_count}_mean_rate"})
        )

    return {
        "jurisdiction_category": (
            targets.groupby(["jurisdiction", "overdose_category"], as_index=False)[TARGET_COLUMN]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": "jurisdiction_category_mean_rate",
                    "std": "jurisdiction_category_std_rate",
                    "min": "jurisdiction_category_min_rate",
                    "max": "jurisdiction_category_max_rate",
                }
            )
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
        "recent_3": recent_mean(3),
        "recent_6": recent_mean(6),
        "recent_12": recent_mean(12),
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
    for period_count in (3, 6, 12):
        output = output.merge(
            features[f"recent_{period_count}"],
            on=["jurisdiction", "overdose_category"],
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
    output["jurisdiction_category_std_rate"] = output["jurisdiction_category_std_rate"].fillna(0.0)
    for column in ("jurisdiction_category_min_rate", "jurisdiction_category_max_rate"):
        output[column] = output[column].fillna(output["jurisdiction_category_mean_rate"])
        output[column] = output[column].fillna(output["category_mean_rate"])
        output[column] = output[column].fillna(global_mean)

    for period_count in (3, 6, 12):
        column = f"recent_{period_count}_mean_rate"
        output[column] = output[column].fillna(output["jurisdiction_category_mean_rate"])
        output[column] = output[column].fillna(output["category_mean_rate"])
        output[column] = output[column].fillna(global_mean)

    return output
