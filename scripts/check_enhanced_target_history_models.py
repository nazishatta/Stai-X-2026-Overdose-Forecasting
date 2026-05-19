"""Experiment 4: enhanced target-history features with HGB/baseline blends."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table
from stai_x_forecasting.evaluation.validation import make_holdout_split, rmse
from stai_x_forecasting.features.target_history import add_target_history_features

TARGET_COLUMN = "rate_per_10000_ed_visits"
CURRENT_BEST_KAGGLE_RMSE = 2.536388
KAGGLE_SCORING_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]

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


def make_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )


def kaggle_rmse(scored: pd.DataFrame) -> float:
    kaggle_scored = scored[scored["overdose_category"].isin(KAGGLE_SCORING_CATEGORIES)]
    return rmse(kaggle_scored[TARGET_COLUMN], kaggle_scored["prediction"])


def category_rmse(scored: pd.DataFrame) -> pd.Series:
    return scored.groupby("overdose_category").apply(
        lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
        include_groups=False,
    )


def print_blend_scores(name: str, scored: pd.DataFrame) -> float:
    score = kaggle_rmse(scored)
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"RMSE Kaggle scoring categories only: {score}")
    print("\nRMSE by overdose_category:")
    print(category_rmse(scored))
    return score


def main() -> None:
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

    model = make_model()
    model.fit(train_table[FEATURES], train_table[TARGET_COLUMN])

    scored = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()
    scored["hgb_prediction"] = model.predict(valid_table[FEATURES]).clip(min=0)
    scored["baseline_prediction"] = valid_table["jurisdiction_category_mean_rate"].clip(lower=0)

    print(f"Train periods: {len(train_periods)}")
    print(f"Validation periods: {len(valid_periods)}")
    print(f"Validation period IDs: {valid_periods}")
    print(f"Train shape with enhanced history features: {train_table.shape}")
    print(f"Validation shape with enhanced history features: {valid_table.shape}")

    blend_scores = {}
    for hgb_weight in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        baseline_weight = 1.0 - hgb_weight
        blend = scored.copy()
        blend["prediction"] = (
            hgb_weight * blend["hgb_prediction"]
            + baseline_weight * blend["baseline_prediction"]
        )
        label = f"{round(hgb_weight * 100)}% HGB + {round(baseline_weight * 100)}% baseline"
        blend_scores[label] = print_blend_scores(label, blend)

    best_label = min(blend_scores, key=blend_scores.get)
    best_score = blend_scores[best_label]
    delta = best_score - CURRENT_BEST_KAGGLE_RMSE

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Current best local Kaggle-category RMSE: {CURRENT_BEST_KAGGLE_RMSE}")
    print(f"Best Experiment 4 blend: {best_label}")
    print(f"Best Experiment 4 Kaggle-category RMSE: {best_score}")
    print(f"Delta vs current best: {delta}")


if __name__ == "__main__":
    main()
