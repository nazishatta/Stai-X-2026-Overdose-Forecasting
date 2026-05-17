"""Submission formatting utilities."""

from __future__ import annotations

import pandas as pd


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: pd.Series,
    prediction_column: str,
) -> pd.DataFrame:
    """Return a submission frame matching Kaggle's sample submission shape."""
    submission = sample_submission.copy()
    submission[prediction_column] = predictions.to_numpy()
    return submission
