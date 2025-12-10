from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.baselines import (
    complete_daily_weekly_blend_predictions,
    daily_weekly_blend_predictions,
    weekly_seasonal_predictions,
)
from taxi_mlops.evaluation import regression_metrics, zone_regression_metrics


def test_weekly_baseline_uses_exact_prior_week_for_same_zone() -> None:
    hours = pd.date_range("2019-01-01", periods=169, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": list(range(len(hours))) + list(range(1000, 1000 + len(hours))),
        }
    )

    predictions = weekly_seasonal_predictions(data)
    target = predictions.loc[
        (predictions["pickup_zone_id"] == 2) & (predictions["pickup_hour"] == hours[-1])
    ].iloc[0]

    assert target["prediction"] == 1000
    assert target["feature_cutoff"] < target["pickup_hour"]


def test_metrics_report_micro_and_macro_errors() -> None:
    predictions = pd.DataFrame(
        {
            "pickup_zone_id": [1, 1, 2],
            "pickup_count": [10, 20, 0],
            "prediction": [12, 18, 4],
        }
    )

    metrics = regression_metrics(predictions)

    assert metrics["mae"] == pytest.approx(8 / 3)
    assert metrics["wape"] == pytest.approx(8 / 30)
    assert metrics["macro_zone_mae"] == pytest.approx(3.0)
    assert metrics["mean_bias"] == pytest.approx(4 / 3)
    assert metrics["zone_mae_max"] == pytest.approx(4.0)
    assert metrics["prediction_coverage"] == 1.0


def test_metrics_reject_incomplete_candidate_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "pickup_zone_id": [1, 1],
            "pickup_count": [10, 20],
            "prediction": [12.0, None],
        }
    )

    with pytest.raises(ValueError, match="no prediction"):
        regression_metrics(predictions)


def test_daily_weekly_blend_uses_only_historical_values() -> None:
    hours = pd.date_range("2019-01-01", periods=169, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "pickup_hour": hours,
            "pickup_zone_id": 1,
            "pickup_count": range(169),
        }
    )

    predictions = daily_weekly_blend_predictions(data, daily_weight=0.25)
    final = predictions.iloc[-1]

    assert final["prediction"] == pytest.approx(0.25 * 144 + 0.75 * 0)
    assert final["feature_cutoff"] < final["pickup_hour"]


def test_zone_metrics_preserve_zone_level_error_and_bias() -> None:
    predictions = pd.DataFrame(
        {
            "pickup_zone_id": [1, 1, 2],
            "pickup_count": [10, 20, 0],
            "prediction": [12, 18, 4],
        }
    )

    zones = zone_regression_metrics(predictions).set_index("pickup_zone_id")

    assert zones.loc[1, "mae"] == pytest.approx(2.0)
    assert zones.loc[1, "mean_bias"] == pytest.approx(0.0)
    assert zones.loc[1, "wape"] == pytest.approx(4 / 30)
    assert zones.loc[2, "mae"] == pytest.approx(4.0)
    assert pd.isna(zones.loc[2, "wape"])


def test_daily_baseline_matches_local_wall_clock_across_spring_dst() -> None:
    data = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(
                [
                    "2019-03-09T07:00:00Z",  # 02:00 EST
                    "2019-03-09T08:00:00Z",  # 03:00 EST
                    "2019-03-10T07:00:00Z",  # 03:00 EDT
                ]
            ),
            "pickup_zone_id": [1, 1, 1],
            "pickup_count": [20, 30, 99],
        }
    )

    predictions = weekly_seasonal_predictions(data, lag_hours=24)
    target = predictions.loc[
        predictions["pickup_hour"] == pd.Timestamp("2019-03-10T07:00:00Z")
    ].iloc[0]

    assert target["prediction"] == 30


def test_complete_blend_falls_back_when_daily_dst_lag_is_missing() -> None:
    data = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(
                [
                    "2019-03-03T07:00:00Z",  # 02:00 EST one week earlier
                    "2019-03-10T06:00:00Z",  # 01:00 EST previous canonical hour
                    "2019-03-10T07:00:00Z",  # 03:00 EDT; daily 03:00 exists too
                    "2019-03-11T06:00:00Z",  # 02:00 EDT; prior-day 02:00 is missing
                ]
            ),
            "pickup_zone_id": [1, 1, 1, 1],
            "pickup_count": [30, 10, 20, 99],
        }
    )

    predictions = complete_daily_weekly_blend_predictions(data)
    target = predictions.loc[
        predictions["pickup_hour"] == pd.Timestamp("2019-03-11T06:00:00Z")
    ].iloc[0]

    assert target["prediction"] == 20
