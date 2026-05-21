"""Experiment 13: scoring-category focused training.

Compare:
A. Current best tuned_raw HGB trained on all 8 overdose categories.
B. Tuned_raw HGB trained only on scoring categories.
C. Tuned_raw HGB trained on all 8 categories, but with sample weights:
   - scoring categories weight = 3.0
   - non-scoring categories weight = 1.0
D. Tuned_raw HGB trained on all 8 categories, but with sample weights:
   - scoring categories weight = 5.0
   - non-scoring categories weight = 1.0

Evaluate only on Kaggle scoring categories.
Report per-fold RMSE, mean RMSE, standard deviation, RMSE by scoring category,
and comparison against current rolling best 2.403409.

Do not create a submission. Do not save models. Do not modify notebooks.
"""

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
CURRENT_ROLLING_BEST = 2.403409

FOLDS = [
    (59, 64),
    (65, 70),
    (71, 76),
]

CATEGORICAL_FEATURES = ["jurisdiction", "overdose_category"]
FEATURES = [
    "jurisdiction",
    "overdose_category",
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


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(model: Any, features: list[str]) -> Pipeline:
    numeric_features = [f for f in features if f not in CATEGORICAL_FEATURES]
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
        "tuned_raw": HistGradientBoostingRegressor(
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_targets = targets[targets["period_id"].isin(train_periods)].copy()
    train_table = training_table[training_table["period_id"].isin(train_periods)].copy()
    valid_table = training_table[training_table["period_id"].isin(validation_periods)].copy()

    train_table = add_target_history_features(train_table, train_targets)
    valid_table = add_target_history_features(valid_table, train_targets)

    valid_table = valid_table[valid_table["overdose_category"].isin(SCORING_CATEGORIES)].copy()
    return train_targets, train_table, valid_table


def build_sample_weights(table: pd.DataFrame, scoring_weight: float) -> np.ndarray:
    weights = np.where(table["overdose_category"].isin(SCORING_CATEGORIES), scoring_weight, 1.0)
    return weights


def fit_and_predict(
    model: Any,
    train_table: pd.DataFrame,
    valid_table: pd.DataFrame,
    features: list[str],
    sample_weight: np.ndarray | None = None,
) -> pd.Series:
    pipeline = make_pipeline(model, features)
    y_train = train_table[TARGET_COLUMN]
    if sample_weight is None:
        pipeline.fit(train_table[features], y_train)
    else:
        pipeline.fit(train_table[features], y_train, model__sample_weight=sample_weight)

    predictions = pipeline.predict(valid_table[features]).clip(min=0)
    return pd.Series(predictions, index=valid_table.index)


def main() -> None:
    dataframes = load_competition_data()
    covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    training_table = clean_training_table(build_training_table(covariates, targets))
    period_order = targets["period_id"].drop_duplicates().tolist()

    models = build_models()
    strategy_labels = [
        "A_tuned_raw_all",
        "B_tuned_raw_scoring_only",
        "C_tuned_raw_weight_3",
        "D_tuned_raw_weight_5",
    ]
    fold_results: dict[str, list[float]] = {label: [] for label in strategy_labels}
    category_accum: dict[str, list[pd.DataFrame]] = {label: [] for label in strategy_labels}

    for fold_index, fold_range in enumerate(FOLDS, start=1):
        train_periods, valid_periods = get_fold_periods(period_order, fold_range)
        train_targets, train_table, valid_table = build_fold_tables(
            training_table,
            targets,
            train_periods,
            valid_periods,
        )

        print("\n" + "=" * 80)
        print(f"Fold {fold_index}: validation period positions {fold_range[0]}–{fold_range[1]}")
        print(f"Validation rows in scoring categories: {valid_table.shape[0]}")

        preds_a = fit_and_predict(models["tuned_raw"], train_table, valid_table, FEATURES)

        train_table_scoring = train_table[train_table["overdose_category"].isin(SCORING_CATEGORIES)].copy()
        preds_b = fit_and_predict(models["tuned_raw"], train_table_scoring, valid_table, FEATURES)

        weights_3 = build_sample_weights(train_table, scoring_weight=3.0)
        preds_c = fit_and_predict(models["tuned_raw"], train_table, valid_table, FEATURES, sample_weight=weights_3)

        weights_5 = build_sample_weights(train_table, scoring_weight=5.0)
        preds_d = fit_and_predict(models["tuned_raw"], train_table, valid_table, FEATURES, sample_weight=weights_5)

        scored_base = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()

        for label, predictions in [
            ("A_tuned_raw_all", preds_a),
            ("B_tuned_raw_scoring_only", preds_b),
            ("C_tuned_raw_weight_3", preds_c),
            ("D_tuned_raw_weight_5", preds_d),
        ]:
            scored = scored_base.copy()
            scored["prediction"] = predictions.values
            fold_rmse = rmse(scored[TARGET_COLUMN], scored["prediction"])
            fold_results[label].append(fold_rmse)
            category_accum[label].append(scored)
            print(f"{label} Fold {fold_index} RMSE: {fold_rmse:.6f}")

    print("\n" + "=" * 80)
    print("Per-fold RMSE")
    print("=" * 80)
    for label, rmses in fold_results.items():
        print(f"{label}: {', '.join(f'{v:.6f}' for v in rmses)}")

    summary = pd.DataFrame(
        {
            "strategy": list(fold_results.keys()),
            "mean_rmse": [np.mean(values) for values in fold_results.values()],
            "std_rmse": [np.std(values, ddof=0) for values in fold_results.values()],
        }
    )

    print("\n" + "=" * 80)
    print("Mean and std across folds")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="{:.6f}".format))

    print("\n" + "=" * 80)
    print("RMSE by scoring category across folds")
    print("=" * 80)
    for label, frames in category_accum.items():
        combined = pd.concat(frames, ignore_index=True)
        category_rmse = combined.groupby("overdose_category").apply(
            lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"])
        )
        print(f"\n{label}")
        print(category_rmse)

    print("\n" + "=" * 80)
    print("Comparison vs current rolling best")
    print("=" * 80)
    for label, values in fold_results.items():
        mean_rmse = np.mean(values)
        print(f"{label} mean RMSE: {mean_rmse:.6f}, delta vs current rolling best: {mean_rmse - CURRENT_ROLLING_BEST:.6f}")


if __name__ == "__main__":
    main()
