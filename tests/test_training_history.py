from __future__ import annotations

import pandas as pd

from taxi_mlops.features import build_feature_table
from taxi_mlops.models import run_training_history_experiment


def test_training_history_uses_common_targets_and_fixed_week_blocks(tmp_path) -> None:
    hours = pd.date_range(
        "2019-01-01", "2020-02-01", freq="h", inclusive="left", tz="UTC"
    )
    data = pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [
                zone * 5 + index % 24 + index % 7
                for zone in (1, 2)
                for index in range(len(hours))
            ],
        }
    )
    features = build_feature_table(data, timezone="UTC")
    strategies = [
        "recent_90d",
        "seasonal_90d_1y_gap",
        "recent_90d_plus_seasonal_90d",
    ]

    results, tiers, weeks = run_training_history_experiment(
        features,
        cutoffs=["2020-01-01T00:00:00Z"],
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h", "lag_168h"],
        strategies=strategies,
        evaluation_days=14,
        timezone="UTC",
        model_parameters={"n_estimators": 5, "n_jobs": 1},
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert set(results["training_history"]) == set(strategies)
    assert results["evaluation_rows"].nunique() == 1
    assert results["prediction_coverage"].eq(1.0).all()
    assert set(tiers["training_history"]) == set(strategies)
    assert weeks.groupby("training_history")["week_index"].nunique().eq(2).all()
