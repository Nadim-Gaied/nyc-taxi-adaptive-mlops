"""Frozen monitoring and promotion decisions used during replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxi_mlops.evaluation import (
    assign_demand_tiers,
    regression_metrics,
    tier_regression_metrics,
    validate_common_targets,
)


@dataclass(frozen=True)
class MonitorDecision:
    weekly_wape: float
    weekly_mean_bias: float
    warning_breach: bool
    severe_breach: bool
    consecutive_warning_weeks: int
    candidate_evaluation_trigger: bool


@dataclass(frozen=True)
class PromotionReference:
    """Candidate-invariant promotion inputs computed once per decision."""

    targets: pd.DataFrame
    current_metrics: dict[str, float | int | None]
    tiers: pd.DataFrame
    current_tiers: pd.DataFrame
    pickup_hour_ns: np.ndarray
    pickup_zone_id: np.ndarray
    pickup_count: np.ndarray
    zone_codes: np.ndarray
    zone_count: int
    tier_codes: np.ndarray
    tier_actual_totals: np.ndarray
    current_tier_wape: np.ndarray


class FrozenPerformanceMonitor:
    def __init__(self, policy: dict[str, object]) -> None:
        monitor = policy["performance_monitor"]
        if not isinstance(monitor, dict):
            raise ValueError("policy is missing performance_monitor")
        self.warning = float(monitor["weekly_wape_warning"])
        self.severe = float(monitor["weekly_wape_severe"])
        self.persistence = int(monitor["persistence_weeks"])
        self._warning_run = 0

    def evaluate(self, predictions: pd.DataFrame) -> MonitorDecision:
        metrics = regression_metrics(predictions)
        wape = float(metrics["wape"])
        mean_bias = float(metrics["mean_bias"])
        warning = wape > self.warning
        severe = wape > self.severe
        self._warning_run = self._warning_run + 1 if warning else 0
        persistent_event = self._warning_run == self.persistence
        return MonitorDecision(
            weekly_wape=wape,
            weekly_mean_bias=mean_bias,
            warning_breach=warning,
            severe_breach=severe,
            consecutive_warning_weeks=self._warning_run,
            candidate_evaluation_trigger=severe or persistent_event,
        )


def prepare_promotion_reference(
    current_predictions: pd.DataFrame,
    tier_reference: pd.DataFrame,
) -> PromotionReference:
    """Precompute the current-model side of a multi-candidate gate."""

    targets = validate_common_targets({"current": current_predictions})
    current_metrics = regression_metrics(current_predictions)
    tiers = assign_demand_tiers(
        tier_reference,
        cumulative_boundaries=(0.80, 0.95),
        tier_labels=("high", "medium", "low"),
    )
    current_tiers = tier_regression_metrics(current_predictions, tiers).set_index("demand_tier")
    zone_codes, zone_values = pd.factorize(targets["pickup_zone_id"], sort=True)
    tier_by_zone = tiers.set_index("pickup_zone_id")["demand_tier"]
    target_tiers = targets["pickup_zone_id"].map(tier_by_zone)
    tier_categories = list(current_tiers.index)
    tier_codes = pd.Categorical(target_tiers, categories=tier_categories).codes.astype(np.int64)
    if (tier_codes < 0).any():
        raise ValueError("current targets contain zones without demand tiers")
    pickup_count = targets["pickup_count"].to_numpy(dtype=float, copy=True)
    tier_actual_totals = np.bincount(
        tier_codes,
        weights=pickup_count,
        minlength=len(tier_categories),
    )
    return PromotionReference(
        targets=targets,
        current_metrics=current_metrics,
        tiers=tiers,
        current_tiers=current_tiers,
        pickup_hour_ns=targets["pickup_hour"].astype("int64").to_numpy(copy=True),
        pickup_zone_id=targets["pickup_zone_id"].to_numpy(copy=True),
        pickup_count=pickup_count,
        zone_codes=zone_codes.astype(np.int64),
        zone_count=len(zone_values),
        tier_codes=tier_codes,
        tier_actual_totals=tier_actual_totals,
        current_tier_wape=current_tiers.loc[tier_categories, "wape"].to_numpy(
            dtype=float,
            copy=True,
        ),
    )


def evaluate_promotion_candidate(
    current_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    tier_reference: pd.DataFrame,
    policy: dict[str, object],
    *,
    prepared_reference: PromotionReference | None = None,
) -> dict[str, object]:
    """Apply every frozen promotion gate to one common holdout window."""

    reference = prepared_reference or prepare_promotion_reference(
        current_predictions, tier_reference
    )
    candidate = candidate_predictions
    candidate_hour_ns = (
        pd.to_datetime(candidate["pickup_hour"], utc=True, errors="raise")
        .astype("int64")
        .to_numpy()
    )
    candidate_zone_id = candidate["pickup_zone_id"].to_numpy()
    candidate_count = candidate["pickup_count"].to_numpy(dtype=float)
    aligned = (
        len(candidate) == len(reference.targets)
        and np.array_equal(candidate_hour_ns, reference.pickup_hour_ns)
        and np.array_equal(candidate_zone_id, reference.pickup_zone_id)
        and np.array_equal(candidate_count, reference.pickup_count)
    )
    if not aligned:
        candidate = candidate.sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(drop=True)
        candidate_hour_ns = (
            pd.to_datetime(candidate["pickup_hour"], utc=True, errors="raise")
            .astype("int64")
            .to_numpy()
        )
        candidate_zone_id = candidate["pickup_zone_id"].to_numpy()
        candidate_count = candidate["pickup_count"].to_numpy(dtype=float)
        aligned = (
            len(candidate) == len(reference.targets)
            and np.array_equal(candidate_hour_ns, reference.pickup_hour_ns)
            and np.array_equal(candidate_zone_id, reference.pickup_zone_id)
            and np.array_equal(candidate_count, reference.pickup_count)
        )
    if not aligned:
        raise ValueError("candidate does not use the common target rows")
    promotion = policy["promotion"]
    demand_tier = policy["demand_tier_monitor"]
    if not isinstance(promotion, dict) or not isinstance(demand_tier, dict):
        raise ValueError("policy is missing promotion or demand-tier settings")
    current_metrics = reference.current_metrics
    prediction = candidate["prediction"].to_numpy(dtype=float)
    if np.isnan(prediction).any():
        raise ValueError("candidate predictions have incomplete coverage")
    absolute_error = np.abs(reference.pickup_count - prediction)
    signed_error = prediction - reference.pickup_count
    actual_total = float(reference.pickup_count.sum())
    zone_error_totals = np.bincount(
        reference.zone_codes,
        weights=absolute_error,
        minlength=reference.zone_count,
    )
    zone_observations = np.bincount(
        reference.zone_codes,
        minlength=reference.zone_count,
    )
    zone_mae = zone_error_totals / zone_observations
    candidate_metrics = {
        "prediction_coverage": 1.0,
        "wape": float(absolute_error.sum() / actual_total),
        "mean_bias": float(signed_error.mean()),
        "zone_mae_p95": float(np.quantile(zone_mae, 0.95)),
    }
    tier_error_totals = np.bincount(
        reference.tier_codes,
        weights=absolute_error,
        minlength=len(reference.tier_actual_totals),
    )
    candidate_tier_wape = tier_error_totals / reference.tier_actual_totals
    tier_regressions = candidate_tier_wape - reference.current_tier_wape

    relative_improvement = float(
        (float(current_metrics["wape"]) - float(candidate_metrics["wape"]))
        / float(current_metrics["wape"])
    )
    maximum_tier_regression = float(tier_regressions.max())
    zone_regression = float(
        float(candidate_metrics["zone_mae_p95"]) - float(current_metrics["zone_mae_p95"])
    )
    bias_regression = float(
        abs(float(candidate_metrics["mean_bias"])) - abs(float(current_metrics["mean_bias"]))
    )
    gates = {
        "coverage_gate_pass": float(candidate_metrics["prediction_coverage"])
        >= float(promotion["required_prediction_coverage"]),
        "wape_improvement_gate_pass": relative_improvement
        >= float(promotion["minimum_relative_wape_improvement"]),
        "tier_regression_gate_pass": maximum_tier_regression
        <= float(promotion["maximum_tier_wape_regression"]),
        "zone_tail_gate_pass": zone_regression
        <= float(promotion["maximum_zone_mae_p95_regression"]),
        "bias_gate_pass": (not bool(promotion["require_bias_not_worse"])) or bias_regression <= 0,
    }
    return {
        "current_wape": float(current_metrics["wape"]),
        "candidate_wape": float(candidate_metrics["wape"]),
        "relative_wape_improvement": relative_improvement,
        "maximum_tier_wape_regression": maximum_tier_regression,
        "zone_mae_p95_regression": zone_regression,
        "absolute_bias_regression": bias_regression,
        **gates,
        "all_quality_gates_pass": all(gates.values()),
    }
