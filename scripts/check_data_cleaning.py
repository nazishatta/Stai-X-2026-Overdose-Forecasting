"""Run cleaning diagnostics without saving data to disk."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import check_duplicates, clean_covariates, clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table


def print_cleaning_report(name: str, before, after, key_columns: list[str]) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Shape before: {before.shape}")
    print(f"Shape after:  {after.shape}")
    print(f"Duplicate count: {check_duplicates(after, key_columns)}")
    print("\nMissing values after cleaning:")
    print(after.isna().sum())


def main() -> None:
    dataframes = load_competition_data()
    train_covariates = dataframes["train/covariates.csv"]
    val_covariates = dataframes["val/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    training_table = build_training_table(train_covariates, targets)

    cleaned_train_covariates = clean_covariates(train_covariates)
    cleaned_val_covariates = clean_covariates(val_covariates)
    cleaned_training_table = clean_training_table(training_table)

    print_cleaning_report(
        "train/covariates.csv",
        train_covariates,
        cleaned_train_covariates,
        ["period_id", "jurisdiction"],
    )
    print_cleaning_report(
        "val/covariates.csv",
        val_covariates,
        cleaned_val_covariates,
        ["period_id", "jurisdiction"],
    )
    print_cleaning_report(
        "merged training table",
        training_table,
        cleaned_training_table,
        ["period_id", "jurisdiction", "overdose_category"],
    )


if __name__ == "__main__":
    main()
