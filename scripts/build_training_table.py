"""Build and inspect the training table without writing data to disk."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import TARGET_COLUMN, build_training_table


def main() -> None:
    dataframes = load_competition_data()
    covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    merged = build_training_table(covariates, targets)

    if TARGET_COLUMN not in merged.columns:
        raise ValueError(f"Target column is missing: {TARGET_COLUMN}")
    if len(merged) != len(targets):
        raise ValueError("Training labels were lost during merge.")

    print(f"Merged shape: {merged.shape}")
    print("\nColumns:")
    print(merged.columns.tolist())
    print("\nMissing values:")
    print(merged.isna().sum())
    print("\nFirst 5 rows:")
    print(merged.head())


if __name__ == "__main__":
    main()
