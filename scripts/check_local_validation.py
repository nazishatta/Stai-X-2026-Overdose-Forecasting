"""Check local holdout validation for the grouped-mean baseline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.evaluation.validation import (
    get_period_order_from_targets,
    make_holdout_split,
    rmse,
)

TARGET_COLUMN = "rate_per_10000_ed_visits"
KEY_COLUMNS = ["jurisdiction", "overdose_category"]
KAGGLE_SCORING_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]


def predict_grouped_mean(train_targets, valid_targets, recent_period_count: int | None = None):
    averaging_targets = train_targets
    if recent_period_count is not None:
        train_period_order = get_period_order_from_targets(train_targets)
        recent_periods = train_period_order[-recent_period_count:]
        averaging_targets = train_targets[train_targets["period_id"].isin(recent_periods)]

    pair_means = (
        averaging_targets.groupby(KEY_COLUMNS, as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "pair_mean"})
    )
    category_means = (
        averaging_targets.groupby("overdose_category", as_index=False)[TARGET_COLUMN]
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


def print_scores(name: str, scored) -> None:
    kaggle_scored = scored[scored["overdose_category"].isin(KAGGLE_SCORING_CATEGORIES)]
    category_rmse = scored.groupby("overdose_category").apply(
        lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
        include_groups=False,
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"RMSE all overdose categories: {rmse(scored[TARGET_COLUMN], scored['prediction'])}")
    print(
        "RMSE Kaggle scoring categories "
        f"({', '.join(KAGGLE_SCORING_CATEGORIES)}): "
        f"{rmse(kaggle_scored[TARGET_COLUMN], kaggle_scored['prediction'])}"
    )
    print("\nRMSE by overdose_category:")
    print(category_rmse)


def main() -> None:
    targets = load_competition_data()["train/dose_sys_train.csv"]
    train_targets, valid_targets, train_periods, valid_periods = make_holdout_split(
        targets,
        n_holdout_periods=6,
    )

    print(f"Train periods: {len(train_periods)}")
    print(f"Validation periods: {len(valid_periods)}")
    print(f"Validation period IDs: {valid_periods}")
    print(f"Train shape: {train_targets.shape}")
    print(f"Validation shape: {valid_targets.shape}")

    baselines = {
        "A. full-history mean by jurisdiction + overdose_category": None,
        "B. recent 3-period mean by jurisdiction + overdose_category": 3,
        "C. recent 6-period mean by jurisdiction + overdose_category": 6,
        "D. recent 12-period mean by jurisdiction + overdose_category": 12,
    }

    for name, recent_period_count in baselines.items():
        scored = predict_grouped_mean(train_targets, valid_targets, recent_period_count)
        print_scores(name, scored)


if __name__ == "__main__":
    main()
