"""Common-target and demand-tier utilities for chronological experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["pickup_hour", "pickup_zone_id", "pickup_count"]


def validate_common_targets(
    candidates: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return canonical targets after proving every candidate uses identical rows."""

    if not candidates:
        raise ValueError("at least one candidate prediction frame is required")

    canonical: pd.DataFrame | None = None
    for candidate_name, predictions in candidates.items():
        missing = set(TARGET_COLUMNS).difference(predictions.columns)
        if missing:
            raise ValueError(
                f"candidate {candidate_name} is missing target columns: "
                f"{', '.join(sorted(missing))}"
            )
        targets = predictions[TARGET_COLUMNS].copy()
        targets["pickup_hour"] = pd.to_datetime(
            targets["pickup_hour"], utc=True, errors="raise"
        )
        if targets.duplicated(["pickup_hour", "pickup_zone_id"]).any():
            raise ValueError(f"candidate {candidate_name} contains duplicate target keys")
        targets = targets.sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(
            drop=True
        )
        if canonical is None:
            canonical = targets
        elif not targets.equals(canonical):
            raise ValueError(
                f"candidate {candidate_name} does not use the common target rows"
            )

    if canonical is None:  # pragma: no cover - guarded by the non-empty check
        raise RuntimeError("common target validation produced no canonical frame")
    return canonical


def assign_demand_tiers(
    training_data: pd.DataFrame,
    *,
    cumulative_boundaries: Sequence[float] = (0.80, 0.95),
    tier_labels: Sequence[str] = ("high", "medium", "low"),
) -> pd.DataFrame:
    """Assign zones by their cumulative share of training-period demand.

    A boundary-crossing zone remains in the preceding tier. Consequently the
    high tier covers at least 80% of training demand with the default policy.
    The function uses training observations only; target-period demand must not
    be supplied here.
    """

    required = {"pickup_zone_id", "pickup_count"}
    missing = required.difference(training_data.columns)
    if missing:
        raise ValueError(
            f"training data is missing columns: {', '.join(sorted(missing))}"
        )
    boundaries = tuple(float(boundary) for boundary in cumulative_boundaries)
    if len(tier_labels) != len(boundaries) + 1:
        raise ValueError("tier_labels must contain one more value than boundaries")
    if any(boundary <= 0 or boundary >= 1 for boundary in boundaries):
        raise ValueError("cumulative boundaries must be strictly between zero and one")
    if tuple(sorted(boundaries)) != boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError("cumulative boundaries must be unique and increasing")
    if training_data["pickup_count"].lt(0).any():
        raise ValueError("pickup_count cannot be negative")

    zones = (
        training_data.groupby("pickup_zone_id", as_index=False)
        .agg(training_pickups=("pickup_count", "sum"))
        .sort_values(["training_pickups", "pickup_zone_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if zones.empty:
        raise ValueError("training data contains no zones")
    total = float(zones["training_pickups"].sum())
    if total <= 0:
        raise ValueError("training data must contain positive total demand")

    zones["demand_rank"] = np.arange(1, len(zones) + 1)
    zones["training_demand_share"] = zones["training_pickups"] / total
    zones["cumulative_demand_share"] = zones["training_demand_share"].cumsum()
    share_before_zone = zones["cumulative_demand_share"] - zones["training_demand_share"]
    tier_index = np.searchsorted(boundaries, share_before_zone, side="right")
    zones["demand_tier"] = [tier_labels[index] for index in tier_index]
    return zones[
        [
            "pickup_zone_id",
            "demand_rank",
            "demand_tier",
            "training_pickups",
            "training_demand_share",
            "cumulative_demand_share",
        ]
    ]


def tier_regression_metrics(
    predictions: pd.DataFrame,
    zone_tiers: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Aggregate forecast quality within precomputed training-demand tiers."""

    required_predictions = {
        "pickup_zone_id",
        "pickup_count",
        "prediction",
    }
    missing_predictions = required_predictions.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(
            "predictions are missing columns: "
            f"{', '.join(sorted(missing_predictions))}"
        )
    required_tiers = {"pickup_zone_id", "demand_tier"}
    missing_tiers = required_tiers.difference(zone_tiers.columns)
    if missing_tiers:
        raise ValueError(
            f"zone tiers are missing columns: {', '.join(sorted(missing_tiers))}"
        )
    if zone_tiers["pickup_zone_id"].duplicated().any():
        raise ValueError("zone tiers contain duplicate pickup_zone_id values")

    missing_count = int(predictions["prediction"].isna().sum())
    if require_complete and missing_count:
        raise ValueError(f"{missing_count} target row(s) have no prediction")
    eligible = predictions.dropna(subset=["pickup_count", "prediction"]).copy()
    eligible = eligible.merge(
        zone_tiers[["pickup_zone_id", "demand_tier"]],
        on="pickup_zone_id",
        how="left",
        validate="many_to_one",
    )
    unassigned = eligible["demand_tier"].isna()
    if unassigned.any():
        missing_zone_ids = sorted(eligible.loc[unassigned, "pickup_zone_id"].unique())
        raise ValueError(f"predictions contain zones without tiers: {missing_zone_ids}")

    eligible["absolute_error"] = (
        eligible["pickup_count"] - eligible["prediction"]
    ).abs()
    eligible["signed_error"] = eligible["prediction"] - eligible["pickup_count"]
    metrics = (
        eligible.groupby("demand_tier", observed=True, as_index=False)
        .agg(
            observations=("pickup_count", "size"),
            zones=("pickup_zone_id", "nunique"),
            actual_pickups=("pickup_count", "sum"),
            predicted_pickups=("prediction", "sum"),
            absolute_error_total=("absolute_error", "sum"),
            mae=("absolute_error", "mean"),
            mean_bias=("signed_error", "mean"),
        )
        .sort_values("demand_tier")
        .reset_index(drop=True)
    )
    metrics["wape"] = metrics["absolute_error_total"].div(
        metrics["actual_pickups"].replace(0, float("nan"))
    )
    metrics["prediction_coverage"] = len(eligible) / len(predictions)
    return metrics
