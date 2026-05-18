"""Check local holdout validation for the grouped-mean baseline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.evaluation.validation import make_holdout_split, rmse

TARGET_COLUMN = "rate_per_10000_ed_visits"
KEY_COLUMNS = ["jurisdiction", "overdose_category"]


def predict_grouped_mean(train_targets, valid_targets):
    pair_means = (
        train_targets.groupby(KEY_COLUMNS, as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "pair_mean"})
    )
    category_means = (
        train_targets.groupby("overdose_category", as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "category_mean"})
    )

    scored = valid_targets.copy()
    scored = scored.merge(pair_means, on=KEY_COLUMNS, how="left", validate="many_to_one")
    scored = scored.merge(
        category_means,
        on="overdose_category",
        how="left",
        validate="many_to_one",
    )
    scored["prediction"] = scored["pair_mean"].fillna(scored["category_mean"])

    if scored["prediction"].isna().any():
        raise ValueError("Local validation predictions contain missing values.")

    return scored


def main() -> None:
    targets = load_competition_data()["train/dose_sys_train.csv"]
    train_targets, valid_targets, train_periods, valid_periods = make_holdout_split(
        targets,
        n_holdout_periods=6,
    )

    scored = predict_grouped_mean(train_targets, valid_targets)
    overall_rmse = rmse(scored[TARGET_COLUMN], scored["prediction"])
    category_rmse = scored.groupby("overdose_category").apply(
        lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
        include_groups=False,
    )

    print(f"Train periods: {len(train_periods)}")
    print(f"Validation periods: {len(valid_periods)}")
    print(f"Validation period IDs: {valid_periods}")
    print(f"Train shape: {train_targets.shape}")
    print(f"Validation shape: {valid_targets.shape}")
    print(f"\nOverall RMSE: {overall_rmse}")
    print("\nRMSE by overdose_category:")
    print(category_rmse)


if __name__ == "__main__":
    main()
