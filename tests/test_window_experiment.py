from __future__ import annotations

import pandas as pd

from taxi_mlops.features import build_feature_table
from taxi_mlops.models import (
    run_feature_ablation_experiment,
    run_seasonal_gap_experiment,
    run_training_window_experiment,
)


def test_window_experiment_keeps_evaluation_targets_fixed() -> None:
    hours = pd.date_range("2019-01-01", periods=24 * 50, freq="h", tz="UTC")
    rows: list[dict[str, object]] = []
    for zone in (1, 2):
        for index, hour in enumerate(hours):
            rows.append(
                {
                    "pickup_hour": hour,
                    "pickup_zone_id": zone,
                    "pickup_count": zone * 5 + index % 24 + index % 7,
                }
            )
    features = build_feature_table(pd.DataFrame(rows), timezone="UTC")

    results, predictions = run_training_window_experiment(
        features,
        cutoffs=["2019-02-10T00:00:00Z"],
        windows_days=(30,),
        include_expanding=True,
        evaluation_days=7,
        model_parameters={"n_estimators": 5, "n_jobs": 1},
    )

    assert set(results["training_window"]) == {"30d", "expanding"}
    counts = predictions.groupby("training_window").size()
    assert counts.nunique() == 1
    assert predictions["prediction"].ge(0).all()
    assert results["prediction_coverage"].eq(1.0).all()


def test_seasonal_gap_experiment_uses_same_targets_for_both_strategies() -> None:
    hours = pd.date_range("2019-01-01", "2020-02-01", freq="h", inclusive="left", tz="UTC")
    data = pd.DataFrame(
        {
            "pickup_hour": hours,
            "pickup_zone_id": 1,
            "pickup_count": [10 + hour.hour + hour.dayofweek for hour in hours],
        }
    )
    features = build_feature_table(data, timezone="UTC")

    results, predictions = run_seasonal_gap_experiment(
        features,
        target_starts=["2020-01-01T00:00:00Z"],
        window_days=90,
        evaluation_days=7,
        timezone="UTC",
        model_parameters={"n_estimators": 5, "n_jobs": 1},
    )

    assert set(results["strategy"]) == {"recent_90d", "seasonal_90d_1y_gap"}
    assert predictions.groupby("strategy").size().nunique() == 1
    assert results["evaluation_start"].nunique() == 1
    assert results["evaluation_end"].nunique() == 1


def test_feature_ablation_reports_training_derived_tiers(tmp_path) -> None:
    hours = pd.date_range("2019-01-01", periods=24 * 50, freq="h", tz="UTC")
    rows: list[dict[str, object]] = []
    for zone in (1, 2):
        for index, hour in enumerate(hours):
            rows.append(
                {
                    "pickup_hour": hour,
                    "pickup_zone_id": zone,
                    "pickup_count": zone * 10 + index % 24,
                }
            )
    features = build_feature_table(pd.DataFrame(rows), timezone="UTC")
    feature_sets = {
        "calendar": ["pickup_zone_id", "hour_of_day", "day_of_week"],
        "calendar_lag": ["pickup_zone_id", "hour_of_day", "day_of_week", "lag_1h"],
    }

    results, tiers = run_feature_ablation_experiment(
        features,
        cutoffs=["2019-02-10T00:00:00Z"],
        feature_sets=feature_sets,
        training_window_days=30,
        evaluation_days=7,
        timezone="UTC",
        model_parameters={"n_estimators": 5, "n_jobs": 1},
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert set(results["feature_set"]) == set(feature_sets)
    assert set(tiers["feature_set"]) == set(feature_sets)
    assert results["prediction_coverage"].eq(1.0).all()
    assert tiers["prediction_coverage"].eq(1.0).all()

    resumed_results, resumed_tiers = run_feature_ablation_experiment(
        features,
        cutoffs=["2019-02-10T00:00:00Z"],
        feature_sets=feature_sets,
        training_window_days=30,
        evaluation_days=7,
        timezone="UTC",
        model_parameters={"n_estimators": 5, "n_jobs": 1},
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert len(resumed_results) == len(results)
    assert len(resumed_tiers) == len(tiers)


def test_training_window_uses_local_calendar_days_across_dst() -> None:
    hours = pd.date_range(
        "2019-02-01T00:00:00Z", "2019-04-01T00:00:00Z", freq="h", inclusive="left"
    )
    data = pd.DataFrame(
        {
            "pickup_hour": hours,
            "pickup_zone_id": 1,
            "pickup_count": [10 + hour.hour for hour in hours],
        }
    )
    features = build_feature_table(data)

    results, _ = run_training_window_experiment(
        features,
        cutoffs=["2019-03-01T00:00:00-05:00"],
        windows_days=(14,),
        include_expanding=False,
        evaluation_days=14,
        timezone="America/New_York",
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h"],
        model_parameters={"n_estimators": 5, "n_jobs": 1},
    )

    # Fourteen local calendar days spanning spring DST contain 335 real hours.
    assert results.iloc[0]["evaluation_rows"] == 14 * 24 - 1
