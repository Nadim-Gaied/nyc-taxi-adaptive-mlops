"""Leakage-safe seasonal forecasting baselines."""

from __future__ import annotations

import pandas as pd


def weekly_seasonal_predictions(
    data: pd.DataFrame,
    *,
    lag_hours: int = 168,
) -> pd.DataFrame:
    """Predict each zone-hour from the same zone exactly ``lag_hours`` earlier."""

    required = {"pickup_hour", "pickup_zone_id", "pickup_count"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {', '.join(sorted(missing))}")
    if lag_hours <= 0:
        raise ValueError("lag_hours must be positive")

    ordered = data.sort_values(["pickup_zone_id", "pickup_hour"]).copy()
    history = ordered[["pickup_hour", "pickup_zone_id", "pickup_count"]].copy()
    history["pickup_hour"] = history["pickup_hour"] + pd.Timedelta(hours=lag_hours)
    history = history.rename(columns={"pickup_count": "prediction"})

    predictions = ordered.merge(
        history,
        on=["pickup_hour", "pickup_zone_id"],
        how="left",
        validate="1:1",
    )
    predictions["model_id"] = f"seasonal_lag_{lag_hours}h"
    predictions["feature_cutoff"] = predictions["pickup_hour"] - pd.Timedelta(hours=1)
    return predictions


def daily_weekly_blend_predictions(
    data: pd.DataFrame,
    *,
    daily_weight: float = 0.5,
) -> pd.DataFrame:
    """Blend lag-24 and lag-168 demand without using target-hour information."""

    if not 0.0 <= daily_weight <= 1.0:
        raise ValueError("daily_weight must be between zero and one")
    weekly_weight = 1.0 - daily_weight
    daily = weekly_seasonal_predictions(data, lag_hours=24).rename(
        columns={"prediction": "daily_prediction"}
    )
    weekly = weekly_seasonal_predictions(data, lag_hours=168)[
        ["pickup_hour", "pickup_zone_id", "prediction"]
    ].rename(columns={"prediction": "weekly_prediction"})
    predictions = daily.merge(
        weekly,
        on=["pickup_hour", "pickup_zone_id"],
        how="left",
        validate="1:1",
    )
    predictions["prediction"] = (
        daily_weight * predictions["daily_prediction"]
        + weekly_weight * predictions["weekly_prediction"]
    )
    predictions["model_id"] = f"daily_weekly_blend_{daily_weight:.2f}_{weekly_weight:.2f}"
    return predictions.drop(columns=["daily_prediction", "weekly_prediction"])
