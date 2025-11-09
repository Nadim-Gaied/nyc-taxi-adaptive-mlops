from __future__ import annotations

import pandas as pd

from taxi_mlops.features import build_feature_table
from taxi_mlops.models import run_training_window_experiment


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

