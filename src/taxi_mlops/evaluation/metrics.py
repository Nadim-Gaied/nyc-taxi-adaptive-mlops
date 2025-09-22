"""Metrics for zone-hour demand forecasts."""

from __future__ import annotations

import pandas as pd


def regression_metrics(
    predictions: pd.DataFrame, *, require_complete: bool = True
) -> dict[str, float | int | None]:
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
