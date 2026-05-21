"""Experiment 11: category-wise blend of tuned_raw and enhanced_log HGB models.

Strategies evaluated per fold (on scoring categories only):
A. tuned_raw for all categories
B. enhanced_log for all categories
C. category-wise selection:
   - all_drugs -> tuned_raw
   - all_opioids -> enhanced_log
   - all_stimulants -> enhanced_log

Reports per-fold RMSEs, mean/std across folds, RMSE by scoring category, and comparison vs current rolling best.
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
        "tuned_raw": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            min_samples_leaf=30,
            random_state=42,
        ),
        "enhanced_log": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
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


def fit_model_and_get_preds(
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


def main() -> None:
    dataframes = load_competition_data()
    covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    training_table = clean_training_table(build_training_table(covariates, targets))
    period_order = targets["period_id"].drop_duplicates().tolist()

    models = build_models()

    strategies = {
        "A_tuned_raw_all": "tuned_raw",
        "B_enhanced_log_all": "enhanced_log",
        "C_categorywise": "categorywise",
    }

    fold_results: dict[str, list[float]] = {name: [] for name in strategies}
    category_accum: dict[str, list[pd.DataFrame]] = {name: [] for name in strategies}

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

        preds_tuned_raw = fit_model_and_get_preds(models["tuned_raw"], train_table, valid_table, FEATURES, log_target=False)
        preds_enhanced_log = fit_model_and_get_preds(models["enhanced_log"], train_table, valid_table, FEATURES, log_target=True)

        scored_base = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()

        # Strategy A: tuned_raw for all
        scored_a = scored_base.copy()
        scored_a["prediction"] = preds_tuned_raw.values
        rmse_a = rmse(scored_a[TARGET_COLUMN], scored_a["prediction"])
        fold_results["A_tuned_raw_all"].append(rmse_a)
        category_accum["A_tuned_raw_all"].append(scored_a)
        print(f"A_tuned_raw_all Fold {fold_index} RMSE: {rmse_a:.6f}")

        # Strategy B: enhanced_log for all
        scored_b = scored_base.copy()
        scored_b["prediction"] = preds_enhanced_log.values
        rmse_b = rmse(scored_b[TARGET_COLUMN], scored_b["prediction"])
        fold_results["B_enhanced_log_all"].append(rmse_b)
        category_accum["B_enhanced_log_all"].append(scored_b)
        print(f"B_enhanced_log_all Fold {fold_index} RMSE: {rmse_b:.6f}")

        # Strategy C: category-wise selection
        scored_c = scored_base.copy()
        # default to tuned_raw then replace for opioid/stimulant
        preds_c = preds_tuned_raw.copy()
        mask_opioid = scored_c["overdose_category"] == "all_opioids"
        mask_stimulant = scored_c["overdose_category"] == "all_stimulants"
        preds_c.loc[mask_opioid] = preds_enhanced_log.loc[mask_opioid]
        preds_c.loc[mask_stimulant] = preds_enhanced_log.loc[mask_stimulant]
        scored_c["prediction"] = preds_c.values
        rmse_c = rmse(scored_c[TARGET_COLUMN], scored_c["prediction"])
        fold_results["C_categorywise"].append(rmse_c)
        category_accum["C_categorywise"].append(scored_c)
        print(f"C_categorywise Fold {fold_index} RMSE: {rmse_c:.6f}")

    print("\n" + "=" * 80)
    print("Per-fold RMSE")
    print("=" * 80)
    for name, rmses in fold_results.items():
        print(f"{name}: {', '.join(f'{v:.6f}' for v in rmses)}")

    summary = pd.DataFrame(
        {
            "strategy": list(fold_results.keys()),
            "mean_rmse": [np.mean(v) for v in fold_results.values()],
            "std_rmse": [np.std(v, ddof=0) for v in fold_results.values()],
        }
    )
    print("\n" + "=" * 80)
    print("Mean and std across folds")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="{:.6f}".format))

    print("\n" + "=" * 80)
    print("RMSE by scoring category across folds")
    print("=" * 80)
    for name, frames in category_accum.items():
        combined = pd.concat(frames, ignore_index=True)
        cat_rmse = combined.groupby("overdose_category").apply(lambda df: rmse(df[TARGET_COLUMN], df["prediction"]))
        print(f"\n{name}")
        print(cat_rmse)

    print("\n" + "=" * 80)
    print("Comparison vs current rolling best")
    print("=" * 80)
    for name, vals in fold_results.items():
        m = np.mean(vals)
        print(f"{name} mean RMSE: {m:.6f}, delta vs current rolling best: {m - CURRENT_ROLLING_BEST:.6f}")


if __name__ == "__main__":
    main()
