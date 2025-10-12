from __future__ import annotations

import pandas as pd

from taxi_mlops.evaluation import rolling_baseline_backtest


def test_rolling_backtest_uses_common_weekly_targets() -> None:
    hours = pd.date_range("2019-01-07", periods=24 * 21, freq="h", tz="UTC")
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

    weekly, summary = rolling_baseline_backtest(pd.DataFrame(rows), timezone="UTC")

    assert weekly["model_id"].nunique() == 4
    assert weekly.groupby("model_id")["observations"].sum().nunique() == 1
    assert weekly["prediction_coverage"].eq(1.0).all()
    assert len(summary["models_ranked"]) == 4
    assert all(item["evaluated_weeks"] == 2 for item in summary["models_ranked"])

