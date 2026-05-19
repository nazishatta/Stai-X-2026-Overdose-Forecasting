"""Experiment 6: tune HGB for enhanced target-history features."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table
from stai_x_forecasting.evaluation.validation import make_holdout_split, rmse
from stai_x_forecasting.features.target_history import add_target_history_features

TARGET_COLUMN = "rate_per_10000_ed_visits"
CURRENT_BEST_LOCAL_RMSE = 2.528977
SCORING_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]

CATEGORICAL_FEATURES = ["jurisdiction", "overdose_category"]
NUMERIC_FEATURES = [
    "unemployment_rate",
    "labor_force",
    "temp_avg_f",
    "precip_in",
    "gtrends_overdose",
    "gtrends_fentanyl",
    "gtrends_naloxone",
    "gtrends_opioid",
    "gtrends_methamphetamine",
    "has_state_doh_release",
    "state_doh_release_length",
    "jurisdiction_category_mean_rate",
    "category_mean_rate",
    "jurisdiction_mean_rate",
    "global_mean_rate",
    "jurisdiction_category_std_rate",
    "jurisdiction_category_min_rate",
    "jurisdiction_category_max_rate",
    "recent_3_mean_rate",
    "recent_6_mean_rate",
    "recent_12_mean_rate",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of grid configurations to run.",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def parameter_grid() -> list[dict[str, float | int]]:
    return [
        {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2_regularization,
            "min_samples_leaf": min_samples_leaf,
        }
        for learning_rate, max_iter, max_leaf_nodes, l2_regularization, min_samples_leaf in product(
            [0.03, 0.05, 0.08, 0.1],
            [100, 200, 300],
            [15, 31, 63],
            [0.0, 0.01, 0.1],
            [10, 20, 30],
        )
    ]


def scoring_rows(frame: pd.DataFrame) -> pd.Series:
    return frame["overdose_category"].isin(SCORING_CATEGORIES)


def main() -> None:
    args = parse_args()
    dataframes = load_competition_data()
    covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    training_table = clean_training_table(build_training_table(covariates, targets))
    train_targets, _, train_periods, valid_periods = make_holdout_split(
        targets,
        n_holdout_periods=6,
    )

    train_table = training_table[training_table["period_id"].isin(train_periods)].copy()
    valid_table = training_table[training_table["period_id"].isin(valid_periods)].copy()
    train_table = add_target_history_features(train_table, train_targets)
    valid_table = add_target_history_features(valid_table, train_targets)

    if train_table[FEATURES].isna().any().any() or valid_table[FEATURES].isna().any().any():
        raise ValueError("Enhanced target-history features contain missing values.")

    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(train_table[FEATURES])
    x_valid = preprocessor.transform(valid_table[FEATURES])
    y_train = train_table[TARGET_COLUMN]
    scoring_mask = scoring_rows(valid_table)
    y_valid_scoring = valid_table.loc[scoring_mask, TARGET_COLUMN]

    grid = parameter_grid()
    if args.max_configs is not None:
        grid = grid[: args.max_configs]

    print(f"Train periods: {len(train_periods)}")
    print(f"Validation periods: {len(valid_periods)}")
    print(f"Validation period IDs: {valid_periods}")
    print(f"Grid configurations: {len(grid)}")

    results = []
    best_model = None
    best_rmse = float("inf")
    best_params = None

    for index, params in enumerate(grid, start=1):
        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid).clip(min=0)
        score = rmse(y_valid_scoring, predictions[scoring_mask.to_numpy()])

        result = dict(params)
        result["rmse"] = score
        results.append(result)

        if score < best_rmse:
            best_rmse = score
            best_params = dict(params)
            best_model = model

        if index == 1 or index % 25 == 0 or index == len(grid):
            print(f"Completed {index}/{len(grid)} configs. Current best RMSE: {best_rmse}")

    results_frame = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    best_predictions = best_model.predict(x_valid).clip(min=0)
    scored = valid_table.loc[scoring_mask, ["overdose_category", TARGET_COLUMN]].copy()
    scored["prediction"] = best_predictions[scoring_mask.to_numpy()]
    category_scores = scored.groupby("overdose_category").apply(
        lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
        include_groups=False,
    )

    print("\n" + "=" * 80)
    print("Top 10 Configurations")
    print("=" * 80)
    print(results_frame.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("Best Model")
    print("=" * 80)
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")
    print("\nRMSE by scoring category for best model:")
    print(category_scores)

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Current best local RMSE: {CURRENT_BEST_LOCAL_RMSE}")
    print(f"Experiment 6 best RMSE: {best_rmse}")
    print(f"Delta vs current best: {best_rmse - CURRENT_BEST_LOCAL_RMSE}")


if __name__ == "__main__":
    main()
