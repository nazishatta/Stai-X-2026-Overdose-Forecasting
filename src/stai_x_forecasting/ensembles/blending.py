"""Prediction blending helpers."""

from __future__ import annotations

import numpy as np


def weighted_average(predictions: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Blend prediction arrays with normalized weights."""
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must have the same length.")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value.")
    normalized = np.asarray(weights, dtype=float) / total_weight
    return np.average(np.vstack(predictions), axis=0, weights=normalized)
