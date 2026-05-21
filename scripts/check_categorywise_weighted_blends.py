"""Experiment 12: category-wise weighted raw/log blends.

For each fold train tuned_raw and enhanced_log HGBs and evaluate category-wise weighted blends.
- all_drugs uses tuned_raw (enhanced weight = 0)
- all_opioids uses enhanced_log weight from [0.5,0.7,0.8,0.9,1.0]
- all_stimulants uses enhanced_log weight from [0.5,0.7,0.8,0.9,1.0]

Reports top 10 weight combinations by mean RMSE, per-fold RMSEs, mean, std, and RMSE by category for best combination.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import itertools
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
CURRENT_ROLLING_BEST = 2.404391

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

OPIOID_WEIGHTS = [0.5, 0.7, 0.8, 0.9, 1.0]
STIMULANT_WEIGHTS = [0.5, 0.7, 0.8, 0.9, 1.0]


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


def build_models() -> Dict[str, Any]:
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


def evaluate_combination(
    tuned_raw_preds: pd.Series,
    enhanced_log_preds: pd.Series,
    valid_table: pd.DataFrame,
    opioid_w: float,
    stimulant_w: float,
) -> float:
    # all_drugs uses tuned_raw only
    preds = tuned_raw_preds.copy()
    mask_opioid = valid_table["overdose_category"] == "all_opioids"
    mask_stimulant = valid_table["overdose_category"] == "all_stimulants"

    preds.loc[mask_opioid] = opioid_w * enhanced_log_preds.loc[mask_opioid] + (1 - opioid_w) * tuned_raw_preds.loc[mask_opioid]
    preds.loc[mask_stimulant] = stimulant_w * enhanced_log_preds.loc[mask_stimulant] + (1 - stimulant_w) * tuned_raw_preds.loc[mask_stimulant]

    return rmse(valid_table[TARGET_COLUMN], preds)


def main() -> None:
    dataframes = load_competition_data()
    covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]

    training_table = clean_training_table(build_training_table(covariates, targets))
    period_order = targets["period_id"].drop_duplicates().tolist()

    models = build_models()

    combos = list(itertools.product(OPIOID_WEIGHTS, STIMULANT_WEIGHTS))
    combo_keys = [(o, s) for (o, s) in combos]

    results: Dict[Tuple[float, float], list[float]] = {k: [] for k in combo_keys}
    combo_frames: Dict[Tuple[float, float], list[pd.DataFrame]] = {k: [] for k in combo_keys}

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

        base = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()

        for opioid_w, stimulant_w in combo_keys:
            rm = evaluate_combination(preds_tuned_raw, preds_enhanced_log, base, opioid_w, stimulant_w)
            results[(opioid_w, stimulant_w)].append(rm)

            # save scored frame for later category RMSE aggregation
            scored = base.copy()
            preds = preds_tuned_raw.copy()
            mask_opioid = scored["overdose_category"] == "all_opioids"
            mask_stimulant = scored["overdose_category"] == "all_stimulants"
            preds.loc[mask_opioid] = opioid_w * preds_enhanced_log.loc[mask_opioid] + (1 - opioid_w) * preds_tuned_raw.loc[mask_opioid]
            preds.loc[mask_stimulant] = stimulant_w * preds_enhanced_log.loc[mask_stimulant] + (1 - stimulant_w) * preds_tuned_raw.loc[mask_stimulant]
            scored["prediction"] = preds.values
            combo_frames[(opioid_w, stimulant_w)].append(scored)

    # summarize combos
    summary_rows = []
    for combo, rms in results.items():
        mean_rmse = np.mean(rms)
        std_rmse = np.std(rms, ddof=0)
        summary_rows.append({"opioid_w": combo[0], "stimulant_w": combo[1], "mean_rmse": mean_rmse, "std_rmse": std_rmse, "per_fold": rms})

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("mean_rmse").reset_index(drop=True)

    top10 = summary_df.head(10)

    print("\n" + "=" * 80)
    print("Top 10 weight combinations by mean RMSE")
    print("=" * 80)
    for idx, row in top10.iterrows():
        print(f"{idx+1}. opioid_w={row.opioid_w}, stimulant_w={row.stimulant_w} -> mean_rmse={row.mean_rmse:.6f}, std={row.std_rmse:.6f}, per_fold={row.per_fold}")

    best = (float(top10.iloc[0].opioid_w), float(top10.iloc[0].stimulant_w))
    print("\nBest combination:", best)

    # RMSE by scoring category for best combo
    best_frames = combo_frames[best]
    combined = pd.concat(best_frames, ignore_index=True)
    cat_rmse = combined.groupby("overdose_category").apply(lambda df: rmse(df[TARGET_COLUMN], df["prediction"]))

    print("\n" + "=" * 80)
    print("RMSE by scoring category for best combination")
    print("=" * 80)
    print(cat_rmse)

    print("\n" + "=" * 80)
    print("Comparison vs current rolling best")
    print("=" * 80)
    print(f"Best mean RMSE: {top10.iloc[0].mean_rmse:.6f}, delta vs current rolling best: {top10.iloc[0].mean_rmse - CURRENT_ROLLING_BEST:.6f}")


if __name__ == "__main__":
    main()
