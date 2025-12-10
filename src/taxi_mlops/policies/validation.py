"""Pre-registered scorecard for lifecycle-policy validation."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import pandas as pd

from taxi_mlops.baselines import complete_daily_weekly_blend_predictions
from taxi_mlops.evaluation import (
    assign_demand_tiers,
    regression_metrics,
    tier_regression_metrics,
    validate_common_targets,
)
from taxi_mlops.models.training_history import _seven_day_metrics


def run_complete_baseline_validation(
    data: pd.DataFrame,
    *,
    cutoffs: Sequence[str | pd.Timestamp],
    evaluation_days: int,
    timezone: str,
    tier_boundaries: Sequence[float],
    tier_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the no-training fallback on the pre-registered target windows."""

    ordered = data.copy()
    ordered["pickup_hour"] = pd.to_datetime(
        ordered["pickup_hour"], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(["pickup_hour", "pickup_zone_id"])
    started = perf_counter()
    all_predictions = complete_daily_weekly_blend_predictions(
        ordered, timezone=timezone
    )
    inference_seconds_total = perf_counter() - started
    result_rows: list[dict[str, object]] = []
    tier_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    for raw_cutoff in cutoffs:
        cutoff = pd.Timestamp(raw_cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("cutoffs must include an explicit UTC offset")
        cutoff = cutoff.tz_convert("UTC")
        evaluation_end = (
            cutoff.tz_convert(timezone) + pd.DateOffset(days=evaluation_days)
        ).tz_convert("UTC")
        predictions = all_predictions.loc[
            all_predictions["pickup_hour"].ge(cutoff)
            & all_predictions["pickup_hour"].lt(evaluation_end)
        ].copy()
        canonical = ordered.loc[
            ordered["pickup_hour"].ge(cutoff)
            & ordered["pickup_hour"].lt(evaluation_end),
            ["pickup_hour", "pickup_zone_id", "pickup_count"],
        ]
        validate_common_targets({"reference": canonical, "baseline": predictions})
        metrics = regression_metrics(predictions)
        result_rows.append(
            {
                "cutoff": str(cutoff),
                "training_history": "daily_weekly_blend_complete",
                "training_intervals": "[]",
                "training_rows": 0,
                "evaluation_start": str(cutoff),
                "evaluation_end": str(evaluation_end),
                "evaluation_rows": int(len(predictions)),
                "training_seconds": 0.0,
                "inference_seconds": float(
                    inference_seconds_total * len(predictions) / len(all_predictions)
                ),
                **metrics,
            }
        )
        tier_reference_start = (
            cutoff.tz_convert(timezone) - pd.DateOffset(days=365)
        ).tz_convert("UTC")
        tier_reference = ordered.loc[
            ordered["pickup_hour"].ge(tier_reference_start)
            & ordered["pickup_hour"].lt(cutoff)
        ]
        tiers = assign_demand_tiers(
            tier_reference,
            cumulative_boundaries=tier_boundaries,
            tier_labels=tier_labels,
        )
        tier_metrics = tier_regression_metrics(predictions, tiers)
        tier_metrics["cutoff"] = str(cutoff)
        tier_metrics["training_history"] = "daily_weekly_blend_complete"
        tier_frames.append(tier_metrics)
        weekly_metrics = _seven_day_metrics(
            predictions, cutoff=cutoff, timezone=timezone
        )
        weekly_metrics["cutoff"] = str(cutoff)
        weekly_metrics["training_history"] = "daily_weekly_blend_complete"
        weekly_frames.append(weekly_metrics)
    return (
        pd.DataFrame(result_rows),
        pd.concat(tier_frames, ignore_index=True),
        pd.concat(weekly_frames, ignore_index=True),
    )


def build_alert_scorecard(
    weekly_results: pd.DataFrame,
    policy: dict[str, object],
    *,
    reference_history: str,
) -> pd.DataFrame:
    """Apply frozen weekly alert thresholds to the reference model.

    Persistence is evaluated within each pre-registered 28-day origin. The
    scorecard therefore measures evaluation pressure; it is not a stateful
    production replay.
    """

    required = {"cutoff", "training_history", "week_index", "week_start", "wape"}
    missing = required.difference(weekly_results.columns)
    if missing:
        raise ValueError(f"weekly results are missing columns: {', '.join(sorted(missing))}")

    monitor = policy["performance_monitor"]
    if not isinstance(monitor, dict):
        raise ValueError("lifecycle policy is missing performance_monitor")
    warning_threshold = float(monitor["weekly_wape_warning"])
    severe_threshold = float(monitor["weekly_wape_severe"])
    persistence_weeks = int(monitor["persistence_weeks"])

    reference = weekly_results.loc[
        weekly_results["training_history"].eq(reference_history)
    ].copy()
    if reference.empty:
        raise ValueError(f"no weekly results for reference history {reference_history}")
    if reference.duplicated(["cutoff", "week_index"]).any():
        raise ValueError("reference weekly results contain duplicate origin-week rows")

    reference = reference.sort_values(["cutoff", "week_index"]).reset_index(drop=True)
    reference["warning_breach"] = reference["wape"].gt(warning_threshold)
    reference["severe_breach"] = reference["wape"].gt(severe_threshold)
    warning_runs: list[int] = []
    persistent_events: list[bool] = []
    for _, origin in reference.groupby("cutoff", sort=False):
        run = 0
        for warning in origin["warning_breach"]:
            run = run + 1 if bool(warning) else 0
            warning_runs.append(run)
            persistent_events.append(run == persistence_weeks)
    reference["consecutive_warning_weeks"] = warning_runs
    reference["persistent_warning_event"] = persistent_events
    reference["candidate_evaluation_trigger"] = (
        reference["severe_breach"] | reference["persistent_warning_event"]
    )
    return reference


def build_promotion_scorecard(
    results: pd.DataFrame,
    tier_results: pd.DataFrame,
    alert_scorecard: pd.DataFrame,
    policy: dict[str, object],
    *,
    reference_history: str,
    candidate_histories: Sequence[str],
) -> pd.DataFrame:
    """Evaluate every frozen promotion gate on common 28-day target windows."""

    required_results = {
        "cutoff",
        "training_history",
        "wape",
        "mean_bias",
        "prediction_coverage",
        "zone_mae_p95",
    }
    missing = required_results.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {', '.join(sorted(missing))}")
    required_tiers = {"cutoff", "training_history", "demand_tier", "wape"}
    missing_tiers = required_tiers.difference(tier_results.columns)
    if missing_tiers:
        raise ValueError(
            f"tier results are missing columns: {', '.join(sorted(missing_tiers))}"
        )

    promotion = policy["promotion"]
    if not isinstance(promotion, dict):
        raise ValueError("lifecycle policy is missing promotion")
    required_coverage = float(promotion["required_prediction_coverage"])
    minimum_improvement = float(promotion["minimum_relative_wape_improvement"])
    maximum_tier_regression = float(promotion["maximum_tier_wape_regression"])
    maximum_zone_regression = float(promotion["maximum_zone_mae_p95_regression"])
    require_bias_not_worse = bool(promotion["require_bias_not_worse"])

    reference = results.loc[results["training_history"].eq(reference_history)].copy()
    if reference.duplicated("cutoff").any():
        raise ValueError("reference results contain duplicate cutoffs")
    reference = reference.set_index("cutoff")
    trigger_by_cutoff = (
        alert_scorecard.groupby("cutoff")["candidate_evaluation_trigger"].any()
    )

    reference_tiers = tier_results.loc[
        tier_results["training_history"].eq(reference_history),
        ["cutoff", "demand_tier", "wape"],
    ].rename(columns={"wape": "reference_tier_wape"})
    rows: list[dict[str, object]] = []
    for candidate_name in candidate_histories:
        candidates = results.loc[results["training_history"].eq(candidate_name)].copy()
        if candidates.empty:
            raise ValueError(f"no results for promotion candidate {candidate_name}")
        candidate_tiers = tier_results.loc[
            tier_results["training_history"].eq(candidate_name),
            ["cutoff", "demand_tier", "wape"],
        ].rename(columns={"wape": "candidate_tier_wape"})
        tier_comparison = candidate_tiers.merge(
            reference_tiers,
            on=["cutoff", "demand_tier"],
            how="inner",
            validate="1:1",
        )
        tier_comparison["tier_wape_regression"] = (
            tier_comparison["candidate_tier_wape"]
            - tier_comparison["reference_tier_wape"]
        )
        maximum_by_cutoff = tier_comparison.groupby("cutoff")[
            "tier_wape_regression"
        ].max()

        for candidate in candidates.itertuples(index=False):
            cutoff = candidate.cutoff
            if cutoff not in reference.index:
                raise ValueError(f"reference history has no result for cutoff {cutoff}")
            current = reference.loc[cutoff]
            relative_improvement = float(
                (float(current["wape"]) - float(candidate.wape))
                / float(current["wape"])
            )
            tier_regression = float(maximum_by_cutoff.loc[cutoff])
            zone_regression = float(
                float(candidate.zone_mae_p95) - float(current["zone_mae_p95"])
            )
            bias_regression = float(
                abs(float(candidate.mean_bias)) - abs(float(current["mean_bias"]))
            )
            coverage_pass = float(candidate.prediction_coverage) >= required_coverage
            wape_pass = relative_improvement >= minimum_improvement
            tier_pass = tier_regression <= maximum_tier_regression
            zone_pass = zone_regression <= maximum_zone_regression
            bias_pass = (not require_bias_not_worse) or bias_regression <= 0
            quality_pass = all(
                (coverage_pass, wape_pass, tier_pass, zone_pass, bias_pass)
            )
            triggered = bool(trigger_by_cutoff.get(cutoff, False))
            rows.append(
                {
                    "cutoff": cutoff,
                    "candidate_history": candidate_name,
                    "candidate_evaluation_triggered": triggered,
                    "relative_wape_improvement": relative_improvement,
                    "maximum_tier_wape_regression": tier_regression,
                    "zone_mae_p95_regression": zone_regression,
                    "absolute_bias_regression": bias_regression,
                    "coverage_gate_pass": coverage_pass,
                    "wape_improvement_gate_pass": wape_pass,
                    "tier_regression_gate_pass": tier_pass,
                    "zone_tail_gate_pass": zone_pass,
                    "bias_gate_pass": bias_pass,
                    "all_quality_gates_pass": quality_pass,
                    "promotion_pressure": triggered and quality_pass,
                }
            )
    return pd.DataFrame(rows).sort_values(["cutoff", "candidate_history"]).reset_index(
        drop=True
    )


def summarize_policy_validation(
    results: pd.DataFrame,
    alert_scorecard: pd.DataFrame,
    promotion_scorecard: pd.DataFrame,
    *,
    reference_history: str,
) -> dict[str, object]:
    """Return a concise, JSON-serializable validation summary."""

    aggregate = (
        results.groupby("training_history")
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_zone_mae_p95=("zone_mae_p95", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
        )
        .sort_values("mean_wape")
        .reset_index()
    )
    return {
        "reference_history": reference_history,
        "origin_count": int(
            results.loc[
                results["training_history"].eq(reference_history), "cutoff"
            ].nunique()
        ),
        "weekly_observations": int(len(alert_scorecard)),
        "warning_weeks": int(alert_scorecard["warning_breach"].sum()),
        "severe_weeks": int(alert_scorecard["severe_breach"].sum()),
        "persistent_warning_events": int(
            alert_scorecard["persistent_warning_event"].sum()
        ),
        "candidate_evaluation_events": int(
            alert_scorecard["candidate_evaluation_trigger"].sum()
        ),
        "triggered_origins": int(
            alert_scorecard.groupby("cutoff")["candidate_evaluation_trigger"].any().sum()
        ),
        "quality_gate_passes": int(promotion_scorecard["all_quality_gates_pass"].sum()),
        "promotion_pressure_events": int(promotion_scorecard["promotion_pressure"].sum()),
        "best_mean_wape_history": str(aggregate.iloc[0]["training_history"]),
        "history_summary": aggregate.to_dict(orient="records"),
    }
