"""Create a grouped-mean baseline Kaggle submission."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.loaders import load_competition_data

TARGET_COLUMN = "rate_per_10000_ed_visits"
KEY_COLUMNS = ["jurisdiction", "overdose_category"]
OUTPUT_PATH = Path("submissions/submission_baseline_mean.csv")


def main() -> None:
    dataframes = load_competition_data()
    train = dataframes["train/dose_sys_train.csv"]
    sample_submission = dataframes["sample_submission.csv"]

    pair_means = (
        train.groupby(KEY_COLUMNS, as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "pair_mean"})
    )
    category_means = (
        train.groupby("overdose_category", as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "category_mean"})
    )

    submission = sample_submission.copy()
    original_row_ids = sample_submission["row_id"].copy()
    submission = submission.merge(pair_means, on=KEY_COLUMNS, how="left", validate="many_to_one")
    submission = submission.merge(
        category_means,
        on="overdose_category",
        how="left",
        validate="many_to_one",
    )
    submission[TARGET_COLUMN] = submission["pair_mean"].fillna(submission["category_mean"])
    submission = submission[sample_submission.columns]

    if len(submission) != len(sample_submission):
        raise ValueError(
            "Submission row count changed: "
            f"expected={len(sample_submission)}, actual={len(submission)}"
        )
    if submission["row_id"].tolist() != original_row_ids.tolist():
        raise ValueError("Submission row_id values do not match sample_submission.csv.")
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("Submission contains missing predictions.")
    if (submission[TARGET_COLUMN] < 0).any():
        raise ValueError("Submission contains negative predictions.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Output path: {OUTPUT_PATH}")
    print(f"Shape: {submission.shape}")
    print("\nFirst 5 rows:")
    print(submission.head())
    print("\nPrediction summary statistics:")
    print(submission[TARGET_COLUMN].describe())


if __name__ == "__main__":
    main()
