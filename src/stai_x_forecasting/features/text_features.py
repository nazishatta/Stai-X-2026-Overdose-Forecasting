"""Keyword-based text features for state_doh_release announcements."""

from __future__ import annotations

import pandas as pd

KEYWORDS = [
    "overdose",
    "opioid",
    "fentanyl",
    "naloxone",
    "methamphetamine",
    "meth",
    "heroin",
    "cocaine",
    "stimulant",
    "emergency",
    "warning",
    "alert",
    "death",
    "fatal",
    "hospital",
    "treatment",
    "recovery",
]


def add_keyword_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add keyword count features from the state_doh_release text.

    The input dataframe is not modified. Missing text values are treated as empty.
    """
    output = df.copy()
    text_series = output.get("state_doh_release", pd.Series(dtype="string")).fillna("").astype(str)
    text_lower = text_series.str.lower()

    for keyword in KEYWORDS:
        output[f"keyword_count_{keyword}"] = text_lower.str.count(keyword)

    keyword_cols = [f"keyword_count_{keyword}" for keyword in KEYWORDS]
    output["total_keyword_count"] = output[keyword_cols].sum(axis=1)
    return output
