from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from taxi_mlops.policies import (
    build_alert_scorecard,
    build_promotion_scorecard,
    load_policy_validation_plan,
    run_complete_baseline_validation,
    summarize_policy_validation,
)


def _policy() -> dict[str, object]:
    return {
        "performance_monitor": {
            "weekly_wape_warning": 0.14,
            "weekly_wape_severe": 0.16,
            "persistence_weeks": 2,
        },
        "promotion": {
            "required_prediction_coverage": 1.0,
            "minimum_relative_wape_improvement": 0.01,
            "maximum_tier_wape_regression": 0.005,
            "maximum_zone_mae_p95_regression": 1.0,
            "require_bias_not_worse": True,
        },
    }


def test_policy_validation_plan_keeps_locked_replay_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    plan = load_policy_validation_plan(
        project_root / "configs/policy_validation_experiment_v1.yaml"
    )

    assert plan["data_access"]["history_splits"] == [
        "development",
        "policy_validation",
    ]
    assert plan["data_access"]["allow_locked_replay"] is False
    assert len(plan["evaluation"]["origins"]) == 6


def test_policy_validation_plan_rejects_locked_replay_access(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "configs/policy_validation_experiment_v1.yaml"
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    plan["data_access"]["allow_locked_replay"] = True
    invalid = tmp_path / "policy_validation.yaml"
    invalid.write_text(yaml.safe_dump(plan), encoding="utf-8")

    try:
        load_policy_validation_plan(invalid)
    except ValueError as error:
        assert "locked replay" in str(error)
    else:  # pragma: no cover
        raise AssertionError("locked replay access should have been rejected")


def test_alert_and_promotion_scorecards_apply_frozen_gates() -> None:
    cutoff = "2019-07-01 04:00:00+00:00"
    weekly = pd.DataFrame(
        {
            "cutoff": [cutoff] * 4,
            "training_history": ["recent_365d"] * 4,
            "week_index": [1, 2, 3, 4],
            "week_start": ["2019-07-01", "2019-07-08", "2019-07-15", "2019-07-22"],
            "wape": [0.15, 0.15, 0.17, 0.13],
        }
    )
    alerts = build_alert_scorecard(weekly, _policy(), reference_history="recent_365d")

    assert alerts["warning_breach"].sum() == 3
    assert alerts["severe_breach"].sum() == 1
    assert alerts["persistent_warning_event"].sum() == 1
    assert alerts["candidate_evaluation_trigger"].sum() == 2

    results = pd.DataFrame(
        [
            {
                "cutoff": cutoff,
                "training_history": "recent_365d",
                "wape": 0.15,
                "mean_bias": 1.0,
                "prediction_coverage": 1.0,
                "zone_mae_p95": 20.0,
                "training_seconds": 2.0,
            },
            {
                "cutoff": cutoff,
                "training_history": "seasonal_90d_1y_gap",
                "wape": 0.145,
                "mean_bias": 0.5,
                "prediction_coverage": 1.0,
                "zone_mae_p95": 20.2,
                "training_seconds": 1.0,
            },
            {
                "cutoff": cutoff,
                "training_history": "expanding",
                "wape": 0.15,
                "mean_bias": 0.5,
                "prediction_coverage": 1.0,
                "zone_mae_p95": 20.0,
                "training_seconds": 3.0,
            },
        ]
    )
    tiers = pd.DataFrame(
        [
            {
                "cutoff": cutoff,
                "training_history": history,
                "demand_tier": tier,
                "wape": value,
            }
            for history, values in {
                "recent_365d": [0.10, 0.20, 0.40],
                "seasonal_90d_1y_gap": [0.099, 0.203, 0.404],
                "expanding": [0.10, 0.20, 0.40],
            }.items()
            for tier, value in zip(["high", "medium", "low"], values, strict=True)
        ]
    )
    promotions = build_promotion_scorecard(
        results,
        tiers,
        alerts,
        _policy(),
        reference_history="recent_365d",
        candidate_histories=["seasonal_90d_1y_gap", "expanding"],
    )

    seasonal = promotions.loc[promotions["candidate_history"].eq("seasonal_90d_1y_gap")].iloc[0]
    expanding = promotions.loc[promotions["candidate_history"].eq("expanding")].iloc[0]
    assert seasonal["all_quality_gates_pass"]
    assert seasonal["promotion_pressure"]
    assert not expanding["wape_improvement_gate_pass"]
    summary = summarize_policy_validation(
        results,
        alerts,
        promotions,
        reference_history="recent_365d",
    )
    assert summary["triggered_origins"] == 1
    assert summary["promotion_pressure_events"] == 1


def test_complete_baseline_validation_has_full_coverage() -> None:
    hours = pd.date_range("2018-01-01", "2019-02-01", freq="h", inclusive="left", tz="UTC")
    data = pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [
                zone * 4 + index % 24 for zone in (1, 2) for index in range(len(hours))
            ],
        }
    )

    results, tiers, weeks = run_complete_baseline_validation(
        data,
        cutoffs=["2019-01-01T00:00:00Z"],
        evaluation_days=14,
        timezone="UTC",
        tier_boundaries=(0.80, 0.95),
        tier_labels=("high", "medium", "low"),
    )

    assert results.iloc[0]["prediction_coverage"] == 1.0
    assert results.iloc[0]["training_rows"] == 0
    assert tiers["training_history"].eq("daily_weekly_blend_complete").all()
    assert weeks["week_index"].nunique() == 2
