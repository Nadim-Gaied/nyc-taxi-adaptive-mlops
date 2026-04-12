"""Leakage-safe seasonal forecasting baselines."""

from __future__ import annotations

import pandas as pd


def weekly_seasonal_predictions(
    data: pd.DataFrame,
    *,
    lag_hours: int = 168,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Predict a zone from its prior seasonal wall-clock observation.

    Daily and weekly lags use New York wall-clock time so that "same hour
    yesterday/week" remains semantically correct across DST changes. Sub-daily
    lags use the prior canonical observations, including across an omitted
    unidentifiable autumn fold hour.
    """

    required = {"pickup_hour", "pickup_zone_id", "pickup_count"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {', '.join(sorted(missing))}")
    if lag_hours <= 0:
        raise ValueError("lag_hours must be positive")

    ordered = data.sort_values(["pickup_zone_id", "pickup_hour"]).copy()
    if lag_hours % 24 == 0:
        ordered["_wall_clock_hour"] = (
            ordered["pickup_hour"].dt.tz_convert(timezone).dt.tz_localize(None)
        )
        history = ordered[["_wall_clock_hour", "pickup_zone_id", "pickup_count"]].copy()
        history["_wall_clock_hour"] = history["_wall_clock_hour"] + pd.Timedelta(hours=lag_hours)
        history = history.rename(columns={"pickup_count": "prediction"})
        predictions = ordered.merge(
            history,
            on=["_wall_clock_hour", "pickup_zone_id"],
            how="left",
            validate="1:1",
        ).drop(columns="_wall_clock_hour")
    else:
        predictions = ordered.copy()
        predictions["prediction"] = predictions.groupby("pickup_zone_id", sort=False)[
            "pickup_count"
        ].shift(lag_hours)
    predictions["model_id"] = f"seasonal_lag_{lag_hours}h"
    predictions["feature_cutoff"] = predictions["pickup_hour"] - pd.Timedelta(hours=1)
    return predictions


def daily_weekly_blend_predictions(
    data: pd.DataFrame,
    *,
    daily_weight: float = 0.5,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Blend lag-24 and lag-168 demand without using target-hour information."""

    if not 0.0 <= daily_weight <= 1.0:
        raise ValueError("daily_weight must be between zero and one")
    weekly_weight = 1.0 - daily_weight
    daily = weekly_seasonal_predictions(data, lag_hours=24, timezone=timezone).rename(
        columns={"prediction": "daily_prediction"}
    )
    weekly = weekly_seasonal_predictions(data, lag_hours=168, timezone=timezone)[
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


def complete_daily_weekly_blend_predictions(
    data: pd.DataFrame,
    *,
    daily_weight: float = 0.5,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Blend daily and weekly lags with a leakage-safe DST fallback chain.

    When one wall-clock lag is unavailable at a DST boundary, use the other.
    If both are unavailable, fall back to the previous canonical hour.
    """

    if not 0.0 <= daily_weight <= 1.0:
        raise ValueError("daily_weight must be between zero and one")

    daily = weekly_seasonal_predictions(data, lag_hours=24, timezone=timezone).rename(
        columns={"prediction": "daily_prediction"}
    )
    weekly = weekly_seasonal_predictions(data, lag_hours=168, timezone=timezone)[
        ["pickup_hour", "pickup_zone_id", "prediction"]
    ].rename(columns={"prediction": "weekly_prediction"})
    previous = weekly_seasonal_predictions(data, lag_hours=1, timezone=timezone)[
        ["pickup_hour", "pickup_zone_id", "prediction"]
    ].rename(columns={"prediction": "previous_prediction"})
    predictions = daily.merge(
        weekly,
        on=["pickup_hour", "pickup_zone_id"],
        how="left",
        validate="1:1",
    ).merge(
        previous,
        on=["pickup_hour", "pickup_zone_id"],
        how="left",
        validate="1:1",
    )
    both_available = (
        predictions["daily_prediction"].notna() & predictions["weekly_prediction"].notna()
    )
    predictions["prediction"] = float("nan")
    predictions.loc[both_available, "prediction"] = (
        daily_weight * predictions.loc[both_available, "daily_prediction"]
        + (1.0 - daily_weight) * predictions.loc[both_available, "weekly_prediction"]
    )
    predictions["prediction"] = (
        predictions["prediction"]
        .fillna(predictions["weekly_prediction"])
        .fillna(predictions["daily_prediction"])
        .fillna(predictions["previous_prediction"])
    )
    predictions["model_id"] = "daily_weekly_blend_complete"
    return predictions.drop(
        columns=["daily_prediction", "weekly_prediction", "previous_prediction"]
    )
