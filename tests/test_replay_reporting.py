import json

import pandas as pd

from taxi_mlops.replay.reporting import write_locked_replay_report


def test_locked_replay_report_writes_weighted_tables_and_figures(tmp_path) -> None:
    strategies = [
        "never_retrain",
        "fixed_periodic_28d",
        "performance_triggered",
        "adaptive_reuse_and_retrain",
    ]
    comparison_rows = []
    for index, strategy in enumerate(strategies):
        strategy_dir = tmp_path / strategy
        strategy_dir.mkdir()
        comparison_rows.append(
            {
                "strategy": strategy,
                "training_jobs": index + 1,
                "training_seconds": 10.0 + index,
                "inference_seconds": 2.0,
                "model_switches": index,
                "registered_models": index + 1,
                "registered_models_evaluated": 100 if strategy.startswith("adaptive") else 0,
                "candidate_evaluation_events": 0,
                "promotion_rejections": 0,
                "degraded_weeks": 2 - (index == 1),
                "consecutive_degraded_weeks_max": 2,
                "observations": 30,
                "actual_pickups": 300.0,
                "predicted_pickups": 295.0,
                "mae": 1.0,
                "wape": 0.20 - 0.01 * index,
                "mean_bias": -0.1,
                "zones": 2,
                "macro_zone_mae": 1.0,
                "zone_mae_p50": 0.8,
                "zone_mae_p90": 1.5,
                "zone_mae_p95": 1.7,
                "zone_mae_max": 2.0,
                "weekly_wape_p90": 0.3,
                "worst_week_wape": 0.4,
            }
        )
        pd.DataFrame(
            {
                "week_index": [1, 2],
                "week_start": ["2020-12-28", "2021-01-04"],
                "observations": [10, 20],
                "actual_pickups": [100.0, 200.0],
                "predicted_pickups": [90.0, 190.0],
                "mae": [2.0, 1.0],
                "wape": [0.30 - 0.01 * index, 0.10 - 0.01 * index],
                "mean_bias": [-1.0, -0.5],
            }
        ).to_csv(strategy_dir / "weekly_metrics.csv", index=False)
        pd.DataFrame(
            {
                "demand_tier": ["low", "medium", "high"],
                "observations": [10, 10, 10],
                "actual_pickups": [30.0, 90.0, 180.0],
                "predicted_pickups": [29.0, 88.0, 178.0],
                "mae": [0.5, 1.0, 1.5],
                "wape": [0.5, 0.2, 0.1],
                "mean_bias": [-0.1, -0.2, -0.2],
            }
        ).to_csv(strategy_dir / "tier_metrics.csv", index=False)
        (strategy_dir / "summary.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(tmp_path / "strategy_comparison.csv", index=False)

    summary = write_locked_replay_report(tmp_path)

    assert summary["status"] == "complete"
    assert summary["observations_per_strategy"] == 30
    output_dir = tmp_path / "analysis"
    annual = pd.read_csv(output_dir / "annual_strategy_metrics.csv")
    never_2020 = annual.loc[annual["strategy"].eq("never_retrain") & annual["year"].eq(2020)].iloc[
        0
    ]
    assert never_2020["wape"] == 0.3
    assert len(list((output_dir / "figures").glob("*.png"))) == 5
    saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["best_overall_wape_strategy"] == "adaptive_reuse_and_retrain"
