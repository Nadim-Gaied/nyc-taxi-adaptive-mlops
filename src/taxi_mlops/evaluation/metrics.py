"""Metrics for zone-hour demand forecasts."""

from __future__ import annotations

import pandas as pd


def _eligible_predictions(predictions: pd.DataFrame, *, require_complete: bool) -> pd.DataFrame:
    required = {"pickup_zone_id", "pickup_count", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {', '.join(sorted(missing))}")

    missing_predictions = int(predictions["prediction"].isna().sum())
    if require_complete and missing_predictions:
        raise ValueError(
            f"{missing_predictions} target row(s) have no prediction; "
            "use a later evaluation start or set require_complete=False"
        )

    eligible = predictions.dropna(subset=["pickup_count", "prediction"]).copy()
    if eligible.empty:
        raise ValueError("no eligible predictions to evaluate")
    return eligible


def regression_metrics(
    predictions: pd.DataFrame, *, require_complete: bool = True
) -> dict[str, float | int | None]:
    eligible = _eligible_predictions(predictions, require_complete=require_complete)

    absolute_error = (eligible["pickup_count"] - eligible["prediction"]).abs()
    signed_error = eligible["prediction"] - eligible["pickup_count"]
    actual_total = float(eligible["pickup_count"].sum())
    zone_mae = absolute_error.groupby(eligible["pickup_zone_id"]).mean()

    return {
        "observations": len(eligible),
        "requested_observations": len(predictions),
        "prediction_coverage": float(len(eligible) / len(predictions)),
        "zones": int(eligible["pickup_zone_id"].nunique()),
        "mae": float(absolute_error.mean()),
        "wape": float(absolute_error.sum() / actual_total) if actual_total else None,
        "mean_bias": float(signed_error.mean()),
        "macro_zone_mae": float(zone_mae.mean()),
        "zone_mae_p50": float(zone_mae.quantile(0.50)),
        "zone_mae_p90": float(zone_mae.quantile(0.90)),
        "zone_mae_p95": float(zone_mae.quantile(0.95)),
        "zone_mae_max": float(zone_mae.max()),
    }


def zone_regression_metrics(
    predictions: pd.DataFrame, *, require_complete: bool = True
) -> pd.DataFrame:
    """Return one diagnostic row per zone for a common evaluation interval."""

    eligible = _eligible_predictions(predictions, require_complete=require_complete)
    eligible["absolute_error"] = (eligible["pickup_count"] - eligible["prediction"]).abs()
    eligible["signed_error"] = eligible["prediction"] - eligible["pickup_count"]
    by_zone = (
        eligible.groupby("pickup_zone_id", as_index=False)
        .agg(
            observations=("pickup_count", "size"),
            actual_pickups=("pickup_count", "sum"),
            predicted_pickups=("prediction", "sum"),
            mean_actual=("pickup_count", "mean"),
            mae=("absolute_error", "mean"),
            absolute_error_total=("absolute_error", "sum"),
            mean_bias=("signed_error", "mean"),
        )
        .sort_values("pickup_zone_id")
    )
    by_zone["wape"] = by_zone["absolute_error_total"].div(
        by_zone["actual_pickups"].replace(0, float("nan"))
    )
    by_zone["prediction_coverage"] = 1.0
    return by_zone
