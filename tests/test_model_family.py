from __future__ import annotations

import pandas as pd

from taxi_mlops.features import build_feature_table
from taxi_mlops.models import run_model_family_experiment


def test_model_family_experiment_uses_complete_common_targets(tmp_path) -> None:
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
    columns = ["pickup_zone_id", "hour_of_day", "day_of_week", "lag_1h"]

    results, tiers = run_model_family_experiment(
        features,
        cutoffs=["2019-02-10T00:00:00Z"],
        feature_columns=columns,
        model_names=[
            "daily_weekly_blend",
            "ridge",
            "poisson_regression",
            "lightgbm_poisson",
        ],
        training_window_days=30,
        evaluation_days=7,
        timezone="UTC",
        model_parameters={
            "poisson_regression": {"max_iter": 100},
            "lightgbm_poisson": {"n_estimators": 5, "n_jobs": 1},
        },
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert set(results["model_name"]) == {
        "daily_weekly_blend",
        "ridge",
        "poisson_regression",
        "lightgbm_poisson",
    }
    assert results["evaluation_rows"].nunique() == 1
    assert results["prediction_coverage"].eq(1.0).all()
    assert set(tiers["model_name"]) == set(results["model_name"])
    assert "hit_iteration_limit" in results.columns
