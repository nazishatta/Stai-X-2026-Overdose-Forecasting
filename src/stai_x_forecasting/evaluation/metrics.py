"""Competition metric helpers."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
