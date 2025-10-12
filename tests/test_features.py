from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.features import build_feature_table


def test_feature_table_uses_only_prior_demand() -> None:
    hours = pd.date_range("2019-01-01", periods=169, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "pickup_hour": hours,
            "pickup_zone_id": 1,
            "pickup_count": range(169),
        }
    )

    features = build_feature_table(data, timezone="UTC")
    final = features.iloc[-1]

    assert final["lag_1h"] == 167
    assert final["lag_24h"] == 144
    assert final["lag_168h"] == 0
    assert final["rolling_mean_24h"] == pytest.approx(sum(range(144, 168)) / 24)
    assert final["rolling_mean_168h"] == pytest.approx(sum(range(168)) / 168)


def test_changing_target_does_not_change_its_features() -> None:
    hours = pd.date_range("2019-01-01", periods=169, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "pickup_hour": hours,
            "pickup_zone_id": 1,
            "pickup_count": range(169),
        }
    )
    changed = data.copy()
    changed.loc[changed.index[-1], "pickup_count"] = 1_000_000

    original_features = build_feature_table(data, timezone="UTC").iloc[-1]
    changed_features = build_feature_table(changed, timezone="UTC").iloc[-1]
    feature_columns = [column for column in data.columns if column != "pickup_count"]
    feature_columns.extend(
        column
        for column in original_features.index
        if column not in data.columns and column not in feature_columns
    )

    pd.testing.assert_series_equal(
        original_features[feature_columns],
        changed_features[feature_columns],
        check_names=False,
    )
