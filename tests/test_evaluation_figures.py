from __future__ import annotations

import pandas as pd

from taxi_mlops.evaluation import (
    write_baseline_backtest_figures,
    write_window_experiment_figures,
)


def test_evaluation_figures_are_written(tmp_path) -> None:
    weekly = pd.DataFrame(
        {
            "model_id": ["daily", "daily", "weekly", "weekly"],
            "evaluation_week": ["2019-W01", "2019-W02", "2019-W01", "2019-W02"],
            "mae": [2.0, 2.2, 1.8, 2.0],
            "zone_mae_p95": [7.0, 7.4, 6.0, 6.4],
        }
    )
    window_results = pd.DataFrame(
        {
            "cutoff": [
                "2019-01-01 00:00:00+00:00",
                "2019-02-01 00:00:00+00:00",
                "2019-01-01 00:00:00+00:00",
                "2019-02-01 00:00:00+00:00",
            ],
            "training_window": ["30d", "30d", "90d", "90d"],
            "mae": [2.1, 2.0, 1.9, 1.8],
            "training_seconds": [1.0, 1.1, 2.0, 2.1],
        }
    )

    baseline_paths = write_baseline_backtest_figures(weekly, tmp_path / "baseline")
    window_paths = write_window_experiment_figures(window_results, tmp_path / "windows")

    assert all(path.stat().st_size > 0 for path in baseline_paths + window_paths)
