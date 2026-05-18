"""Compare tabular models with historical target aggregate features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table
from stai_x_forecasting.evaluation.validation import make_holdout_split, rmse
from stai_x_forecasting.features.target_history import add_target_history_features

TARGET_COLUMN = "rate_per_10000_ed_visits"
PREVIOUS_BEST_KAGGLE_RMSE = 2.735459
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


def score_predictions(scored: pd.DataFrame) -> tuple[float, float, pd.Series]:
    kaggle_scored = scored[scored["overdose_category"].isin(KAGGLE_SCORING_CATEGORIES)]
    category_rmse = scored.groupby("overdose_category").apply(
        lambda frame: rmse(frame[TARGET_COLUMN], frame["prediction"]),
        include_groups=False,
    )
    return (
        rmse(scored[TARGET_COLUMN], scored["prediction"]),
        rmse(kaggle_scored[TARGET_COLUMN], kaggle_scored["prediction"]),
        category_rmse,
    )


def print_scores(model_name: str, scored: pd.DataFrame) -> float:
    overall_rmse, kaggle_rmse, category_rmse = score_predictions(scored)

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print(f"RMSE all overdose categories: {overall_rmse}")
    print(
        "RMSE Kaggle scoring categories "
        f"({', '.join(KAGGLE_SCORING_CATEGORIES)}): {kaggle_rmse}"
    )
    print("\nRMSE by overdose_category:")
    print(category_rmse)
    return kaggle_rmse


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

    x_train = train_table[FEATURES]
    y_train = train_table[TARGET_COLUMN]
    x_valid = valid_table[FEATURES]

    print(f"Train periods: {len(train_periods)}")
    print(f"Validation periods: {len(valid_periods)}")
    print(f"Validation period IDs: {valid_periods}")
    print(f"Train shape with history features: {train_table.shape}")
    print(f"Validation shape with history features: {valid_table.shape}")

    models = {
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
    }

    scored_predictions = {}
    for model_name, model in models.items():
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)

        scored = valid_table[["period_id", "jurisdiction", "overdose_category", TARGET_COLUMN]].copy()
        scored["prediction"] = pipeline.predict(x_valid)
        scored["prediction"] = scored["prediction"].clip(lower=0)

        scored_predictions[model_name] = scored
        print_scores(model_name, scored)

    hybrid = scored_predictions["HistGradientBoostingRegressor"].copy()
    extra_trees_predictions = scored_predictions["ExtraTreesRegressor"]["prediction"]
    extra_trees_categories = hybrid["overdose_category"].isin(["all_opioids", "all_stimulants"])
    hybrid.loc[extra_trees_categories, "prediction"] = extra_trees_predictions[extra_trees_categories]

    hybrid_kaggle_rmse = print_scores("Category-specific hybrid", hybrid)
    delta = hybrid_kaggle_rmse - PREVIOUS_BEST_KAGGLE_RMSE
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Previous best local Kaggle-category RMSE: {PREVIOUS_BEST_KAGGLE_RMSE}")
    print(f"Target-history hybrid Kaggle-category RMSE: {hybrid_kaggle_rmse}")
    print(f"Delta vs previous best: {delta}")


if __name__ == "__main__":
    main()
