"""Experiment 9: blend original target-history HGB with tuned enhanced HGB."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table
from stai_x_forecasting.evaluation.metrics import rmse
from stai_x_forecasting.features.target_history import add_target_history_features

TARGET_COLUMN = "rate_per_10000_ed_visits"
SCORING_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]
CURRENT_ROLLING_BEST = 2.415053

FOLDS = [
    (59, 64),
    (65, 70),
    (71, 76),
]

CATEGORICAL_FEATURES = ["jurisdiction", "overdose_category"]
ORIGINAL_NUMERIC_FEATURES = [
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
]
TUNED_NUMERIC_FEATURES = ORIGINAL_NUMERIC_FEATURES + [
    "jurisdiction_category_std_rate",
    "jurisdiction_category_min_rate",
    "jurisdiction_category_max_rate",
    "recent_3_mean_rate",
    "recent_6_mean_rate",
    "recent_12_mean_rate",
]
ORIGINAL_FEATURES = CATEGORICAL_FEATURES + ORIGINAL_NUMERIC_FEATURES
TUNED_FEATURES = CATEGORICAL_FEATURES + TUNED_NUMERIC_FEATURES

BLEND_WEIGHTS = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(model: Any, numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_fold_periods(period_order: list[str], fold: tuple[int, int]) -> tuple[list[str], list[str]]:
    start, end = fold
    if not (0 <= start <= end < len(period_order)):
        raise ValueError(f"Fold positions are invalid: {fold}")
    validation_periods = period_order[start : end + 1]
    training_periods = period_order[:start]
    return training_periods, validation_periods


def build_models() -> dict[str, Any]:
    return {
        "original": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
        "tuned": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            min_samples_leaf=30,
            random_state=42,
        ),
    }


def build_fold_tables(
    training_table: pd.DataFrame,
    targets: pd.DataFrame,
    train_periods: list[str],
    validation_periods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_targets = targets[targets["period_id"].isin(train_periods)].copy()

    train_table = training_table[training_table["period_id"].isin(train_periods)].copy()
    valid_table = training_table[training_table["period_id"].isin(validation_periods)].copy()

    train_table = add_target_history_features(train_table, train_targets)
    valid_table = add_target_history_features(valid_table, train_targets)
    valid_table = valid_table[valid_table["overdose_category"].isin(SCORING_CATEGORIES)].copy()

    return train_table, valid_table


def predict_fold(
    train_table: pd.DataFrame,
    valid_table: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    models = build_models()

    original_pipeline = make_pipeline(models["original"], ORIGINAL_NUMERIC_FEATURES)
    original_pipeline.fit(train_table[ORIGINAL_FEATURES], train_table[TARGET_COLUMN])
    original_pred = original_pipeline.predict(valid_table[ORIGINAL_FEATURES]).clip(min=0)

    tuned_pipeline = make_pipeline(models["tuned"], TUNED_NUMERIC_FEATURES)
    tuned_pipeline.fit(train_table[TUNED_FEATURES], train_table[TARGET_COLUMN])
    tuned_pred = tuned_pipeline.predict(valid_table[TUNED_FEATURES]).clip(min=0)

    return pd.Series(original_pred, index=valid_table.index), pd.Series(tuned_pred, index=valid_table.index)


def score_blends(
    valid_table: pd.DataFrame,
    original_pred: pd.Series,
    tuned_pred: pd.Series,
) -> dict[float, tuple[float, pd.Series]]:
    results: dict[float, tuple[float, pd.Series]] = {}
    truth = valid_table[TARGET_COLUMN]
    for weight in BLEND_WEIGHTS:
        prediction = weight * original_pred + (1.0 - weight) * tuned_pred
        rmse_value = rmse(truth, prediction)
        results[weight] = (rmse_value, prediction)
    return results


def summarize_results(results: dict[float, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "blend_weight": list(results.keys()),
            "mean_rmse": [np.mean(values) for values in results.values()],
            "std_rmse": [np.std(values, ddof=0) for values in results.values()],
        }
    )


def main() -> None:
    dataframes = load_competition_data()
    covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    training_table = clean_training_table(build_training_table(covariates, targets))
    period_order = targets["period_id"].drop_duplicates().tolist()

    fold_scores: dict[float, list[float]] = {weight: [] for weight in BLEND_WEIGHTS}
    category_scores: dict[float, list[pd.Series]] = {weight: [] for weight in BLEND_WEIGHTS}

    for fold_index, fold_range in enumerate(FOLDS, start=1):
        train_periods, valid_periods = get_fold_periods(period_order, fold_range)
        train_table, valid_table = build_fold_tables(
            training_table,
            targets,
            train_periods,
            valid_periods,
        )

        original_pred, tuned_pred = predict_fold(train_table, valid_table)
        fold_results = score_blends(valid_table, original_pred, tuned_pred)

        print("\n" + "=" * 80)
        print(f"Fold {fold_index}: validation period positions {fold_range[0]}–{fold_range[1]}")
        print(f"Training periods: {len(train_periods)}")
        print(f"Validation periods: {len(valid_periods)}")
        print(f"Validation period IDs: {valid_periods}")
        print(f"Validation rows in scoring categories: {valid_table.shape[0]}")

        for weight, (rmse_value, prediction) in fold_results.items():
            fold_scores[weight].append(rmse_value)
            scored = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()
            scored["prediction"] = prediction
            category_rmse = scored.groupby("overdose_category").apply(
                lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
                include_groups=False,
            )
            category_scores[weight].append(category_rmse)
            print(f"Blend {weight:.2f} original / {1-weight:.2f} tuned RMSE: {rmse_value:.6f}")

    print("\n" + "=" * 80)
    print("Fold RMSE by blend")
    print("=" * 80)
    for weight, rmses in fold_scores.items():
        print(f"{weight:.2f}: {', '.join(f'{value:.6f}' for value in rmses)}")

    summary = summarize_results(fold_scores)
    print("\n" + "=" * 80)
    print("Mean and standard deviation across folds")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="{:.6f}".format))

    best_weight = summary.loc[summary["mean_rmse"].idxmin(), "blend_weight"]
    best_mean = summary["mean_rmse"].min()
    print("\n" + "=" * 80)
    print("Best blend")
    print("=" * 80)
    print(f"Best blend weight: {best_weight:.2f}")
    print(f"Best mean RMSE: {best_mean:.6f}")
    print(f"Current rolling best: {CURRENT_ROLLING_BEST:.6f}")

    print("\n" + "=" * 80)
    print("RMSE by scoring category across folds for best blend")
    print("=" * 80)
    best_categories = pd.concat(category_scores[best_weight], axis=1)
    best_categories.columns = [f"fold_{i+1}" for i in range(len(best_categories.columns))]
    print(best_categories.mean(axis=1).to_string())

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    public_rmse = np.mean(fold_scores[1.0])
    print(f"Public-best model rolling RMSE: {public_rmse:.6f}")
    print(f"Current rolling best: {CURRENT_ROLLING_BEST:.6f}")
    print(f"Delta from current rolling best: {best_mean - CURRENT_ROLLING_BEST:.6f}")


if __name__ == "__main__":
    main()
