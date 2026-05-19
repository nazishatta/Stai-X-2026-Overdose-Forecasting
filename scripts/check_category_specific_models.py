"""Experiment 5: category-specific models with enhanced target-history features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table
from stai_x_forecasting.evaluation.validation import make_holdout_split, rmse
from stai_x_forecasting.features.target_history import add_target_history_features

TARGET_COLUMN = "rate_per_10000_ed_visits"
CURRENT_BEST_KAGGLE_RMSE = 2.528977
SCORING_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]

CATEGORICAL_FEATURES = ["jurisdiction"]
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


def make_pipeline(model) -> Pipeline:
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
            ("model", model),
        ]
    )


def model_candidates() -> dict[str, object]:
    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


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

    train_table = train_table[train_table["overdose_category"].isin(SCORING_CATEGORIES)].copy()
    valid_table = valid_table[valid_table["overdose_category"].isin(SCORING_CATEGORIES)].copy()

    if train_table[FEATURES].isna().any().any() or valid_table[FEATURES].isna().any().any():
        raise ValueError("Category-specific features contain missing values.")

    print(f"Train periods: {len(train_periods)}")
    print(f"Validation periods: {len(valid_periods)}")
    print(f"Validation period IDs: {valid_periods}")
    print(f"Training rows in scoring categories: {train_table.shape}")
    print(f"Validation rows in scoring categories: {valid_table.shape}")

    combined_predictions = []
    best_models = {}

    for category in SCORING_CATEGORIES:
        category_train = train_table[train_table["overdose_category"] == category].copy()
        category_valid = valid_table[valid_table["overdose_category"] == category].copy()

        print("\n" + "=" * 80)
        print(category)
        print("=" * 80)

        category_scores = {}
        category_predictions = {}
        for model_name, model in model_candidates().items():
            pipeline = make_pipeline(model)
            pipeline.fit(category_train[FEATURES], category_train[TARGET_COLUMN])

            scored = category_valid[
                ["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]
            ].copy()
            scored["prediction"] = pipeline.predict(category_valid[FEATURES]).clip(min=0)
            model_rmse = rmse(scored[TARGET_COLUMN], scored["prediction"])

            category_scores[model_name] = model_rmse
            category_predictions[model_name] = scored
            print(f"{model_name} RMSE: {model_rmse}")

        best_model_name = min(category_scores, key=category_scores.get)
        best_models[category] = best_model_name
        combined_predictions.append(category_predictions[best_model_name])
        print(f"Best model for {category}: {best_model_name}")

    combined = pd.concat(combined_predictions, ignore_index=True)
    combined_rmse = rmse(combined[TARGET_COLUMN], combined["prediction"])
    combined_category_rmse = combined.groupby("overdose_category").apply(
        lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
        include_groups=False,
    )
    delta = combined_rmse - CURRENT_BEST_KAGGLE_RMSE

    print("\n" + "=" * 80)
    print("Combined Category-Specific Prediction")
    print("=" * 80)
    print(f"Best models by category: {best_models}")
    print(f"RMSE on Kaggle scoring categories only: {combined_rmse}")
    print("\nRMSE by scoring category:")
    print(combined_category_rmse)

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Current best local Kaggle-category RMSE: {CURRENT_BEST_KAGGLE_RMSE}")
    print(f"Experiment 5 Kaggle-category RMSE: {combined_rmse}")
    print(f"Delta vs current best: {delta}")


if __name__ == "__main__":
    main()
