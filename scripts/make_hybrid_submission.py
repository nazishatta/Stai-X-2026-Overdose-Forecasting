"""Create a Kaggle submission from the category-specific hybrid tabular model."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stai_x_forecasting.data.cleaning import clean_covariates, clean_training_table
from stai_x_forecasting.data.loaders import load_competition_data
from stai_x_forecasting.data.preprocessing import build_training_table

TARGET_COLUMN = "rate_per_10000_ed_visits"
OUTPUT_COLUMNS = ["row_id", TARGET_COLUMN]
OUTPUT_PATH = Path("submissions/submission_hybrid_tabular.csv")

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


def main() -> None:
    dataframes = load_competition_data()
    train_covariates = dataframes["train/covariates.csv"]
    targets = dataframes["train/dose_sys_train.csv"]
    val_covariates = dataframes["val/covariates.csv"]
    sample_submission = dataframes["sample_submission.csv"]

    training_table = clean_training_table(build_training_table(train_covariates, targets))
    cleaned_val_covariates = clean_covariates(val_covariates)

    prediction_table = sample_submission.drop(columns=[TARGET_COLUMN]).merge(
        cleaned_val_covariates,
        on=["period_id", "jurisdiction"],
        how="left",
        validate="many_to_one",
    )

    if len(prediction_table) != len(sample_submission):
        raise ValueError(
            "Prediction table row count changed: "
            f"expected={len(sample_submission)}, actual={len(prediction_table)}"
        )
    if prediction_table[FEATURES].isna().any().any():
        raise ValueError("Prediction features contain missing values after merging val covariates.")

    x_train = training_table[FEATURES]
    y_train = training_table[TARGET_COLUMN]
    x_predict = prediction_table[FEATURES]

    hist_model = make_pipeline(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
    )
    extra_trees_model = make_pipeline(
        ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )

    hist_model.fit(x_train, y_train)
    extra_trees_model.fit(x_train, y_train)

    submission = sample_submission[["row_id", "overdose_category"]].copy()
    submission["hist_prediction"] = hist_model.predict(x_predict)
    submission["extra_trees_prediction"] = extra_trees_model.predict(x_predict)
    submission[TARGET_COLUMN] = submission["hist_prediction"]

    extra_trees_categories = submission["overdose_category"].isin(["all_opioids", "all_stimulants"])
    submission.loc[extra_trees_categories, TARGET_COLUMN] = submission.loc[
        extra_trees_categories,
        "extra_trees_prediction",
    ]
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].clip(lower=0)
    submission = submission[OUTPUT_COLUMNS]

    if len(submission) != 918:
        raise ValueError(f"Submission must have exactly 918 rows, found {len(submission)}.")
    if submission.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {OUTPUT_COLUMNS}.")
    if submission["row_id"].tolist() != sample_submission["row_id"].tolist():
        raise ValueError("Submission row_id values do not match sample_submission.csv.")
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("Submission contains missing predictions.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Output path: {OUTPUT_PATH}")
    print(f"Shape: {submission.shape}")
    print(f"Columns: {submission.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(submission.head())
    print("\nPrediction summary:")
    print(submission[TARGET_COLUMN].describe())
    print(f"\nMissing prediction count: {submission[TARGET_COLUMN].isna().sum()}")
    print(f"Min prediction: {submission[TARGET_COLUMN].min()}")
    print(f"Max prediction: {submission[TARGET_COLUMN].max()}")


if __name__ == "__main__":
    main()
