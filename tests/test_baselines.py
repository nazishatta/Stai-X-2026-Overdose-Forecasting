from __future__ import annotations

import pandas as pd

from stai_x_forecasting.models.baselines import GroupMeanForecaster, LastValueForecaster


def test_last_value_forecaster_uses_latest_group_value() -> None:
    train = pd.DataFrame({"state": ["A", "A", "B"], "target": [1.0, 3.0, 10.0]})
    test = pd.DataFrame({"state": ["A", "B", "C"]})

    model = LastValueForecaster(group_column="state", target_column="target").fit(train)

    assert model.predict(test).tolist() == [3.0, 10.0, train["target"].mean()]


def test_group_mean_forecaster_uses_group_average() -> None:
    train = pd.DataFrame({"state": ["A", "A", "B"], "target": [1.0, 3.0, 10.0]})
    test = pd.DataFrame({"state": ["A", "B"]})

    model = GroupMeanForecaster(group_column="state", target_column="target").fit(train)

    assert model.predict(test).tolist() == [2.0, 10.0]
