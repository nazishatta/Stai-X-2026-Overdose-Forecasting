from __future__ import annotations

import numpy as np

from stai_x_forecasting.evaluation.metrics import rmse


def test_rmse_zero_for_identical_arrays() -> None:
    values = np.array([1.0, 2.0, 3.0])

    assert rmse(values, values) == 0.0
