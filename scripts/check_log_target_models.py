"""Experiment 10: compare raw-target vs log-target HGB models under rolling validation.

Compares:
- Enhanced Target-History HGB trained on raw target
- Enhanced Target-History HGB trained on log1p(target) then expm1-inverted
- Tuned HGB trained on raw target
- Tuned HGB trained on log1p(target) then expm1-inverted

Reports RMSE on Kaggle scoring categories only, mean/std across folds, RMSE by category.
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
CURRENT_ROLLING_BEST = 2.415053

FOLDS = [
    (59, 64),
    (65, 70),
    (71, 76),
]

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
        "enhanced_raw": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
        "enhanced_log": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
        "tuned_raw": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            min_samples_leaf=30,
            random_state=42,
        ),
        "tuned_log": HistGradientBoostingRegressor(
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


def fit_and_predict(
    model: Any,
    train_table: pd.DataFrame,
    valid_table: pd.DataFrame,
    features: list[str],
    log_target: bool = False,
) -> pd.Series:
    pipeline = make_pipeline(model, features)
    if log_target:
        y_train = np.log1p(train_table[TARGET_COLUMN])
        pipeline.fit(train_table[features], y_train)
        pred_log = pipeline.predict(valid_table[features])
        preds = np.expm1(pred_log)
    else:
        y_train = train_table[TARGET_COLUMN]
        pipeline.fit(train_table[features], y_train)
        preds = pipeline.predict(valid_table[features])

    preds = np.maximum(preds, 0.0)
    return pd.Series(preds, index=valid_table.index)


def summarize_results(results: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": list(results.keys()),
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

    models = build_models()
    model_names = list(models.keys())

    fold_results: dict[str, list[float]] = {name: [] for name in model_names}
    category_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in model_names}

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
        print(f"Training periods: {len(train_periods)}")
        print(f"Validation periods: {len(valid_periods)}")
        print(f"Validation period IDs: {valid_periods}")
        print(f"Validation rows in scoring categories: {valid_table.shape[0]}")

        # Enhanced HGB on raw target
        preds_enhanced_raw = fit_and_predict(models["enhanced_raw"], train_table, valid_table, FEATURES, log_target=False)
        # Enhanced HGB on log target
        preds_enhanced_log = fit_and_predict(models["enhanced_log"], train_table, valid_table, FEATURES, log_target=True)
        # Tuned HGB on raw target
        preds_tuned_raw = fit_and_predict(models["tuned_raw"], train_table, valid_table, FEATURES, log_target=False)
        # Tuned HGB on log target
        preds_tuned_log = fit_and_predict(models["tuned_log"], train_table, valid_table, FEATURES, log_target=True)

        preds_map = {
            "enhanced_raw": preds_enhanced_raw,
            "enhanced_log": preds_enhanced_log,
            "tuned_raw": preds_tuned_raw,
            "tuned_log": preds_tuned_log,
        }

        for name, preds in preds_map.items():
            scored = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()
            scored["prediction"] = preds.values
            model_rmse = rmse(scored[TARGET_COLUMN], scored["prediction"])
            fold_results[name].append(model_rmse)
            category_predictions[name].append(scored)
            print(f"{name} Fold {fold_index} RMSE: {model_rmse:.6f}")

    print("\n" + "=" * 80)
    print("Per-fold RMSE")
    print("=" * 80)
    for model_name, rmses in fold_results.items():
        print(f"{model_name}: {', '.join(f'{value:.6f}' for value in rmses)}")

    summary = summarize_results(fold_results)
    print("\n" + "=" * 80)
    print("Mean and standard deviation across folds")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="{:.6f}".format))

    print("\n" + "=" * 80)
    print("RMSE by scoring category across folds")
    print("=" * 80)
    for model_name, scored_frames in category_predictions.items():
        combined = pd.concat(scored_frames, ignore_index=True)
        category_rmse = combined.groupby("overdose_category").apply(
            lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
            include_groups=False,
        )
        print(f"\n{model_name}")
        print(category_rmse)

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    for model_name, values in fold_results.items():
        mean_rmse = np.mean(values)
        print(f"{model_name} mean RMSE: {mean_rmse:.6f}, delta vs current rolling best: {mean_rmse - CURRENT_ROLLING_BEST:.6f}")


if __name__ == "__main__":
    main()
