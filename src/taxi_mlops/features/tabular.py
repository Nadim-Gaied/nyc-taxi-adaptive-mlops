"""Tabular calendar, lag, and rolling features for hourly zone demand."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxi_mlops.baselines import weekly_seasonal_predictions


def build_feature_table(
    data: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
    lags: tuple[int, ...] = (1, 24, 168),
    rolling_windows: tuple[int, ...] = (24, 168),
) -> pd.DataFrame:
    """Build features for each target hour using only earlier demand values."""

    required = {"pickup_hour", "pickup_zone_id", "pickup_count"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {', '.join(sorted(missing))}")
    if any(lag <= 0 for lag in lags):
        raise ValueError("all lags must be positive")
    if any(window <= 0 for window in rolling_windows):
        raise ValueError("all rolling windows must be positive")

    features = data[list(required)].copy()
    features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], utc=True, errors="raise")
    features = features.sort_values(["pickup_zone_id", "pickup_hour"]).reset_index(drop=True)
    if features.duplicated(["pickup_hour", "pickup_zone_id"]).any():
        raise ValueError("canonical data contains duplicate zone-hour keys")

    local_time = features["pickup_hour"].dt.tz_convert(timezone)
    features["hour_of_day"] = local_time.dt.hour.astype("int8")
    features["day_of_week"] = local_time.dt.dayofweek.astype("int8")
    features["month"] = local_time.dt.month.astype("int8")
    features["day_of_year"] = local_time.dt.dayofyear.astype("int16")
    features["is_weekend"] = features["day_of_week"].ge(5).astype("int8")
    features["hour_sin"] = np.sin(2 * np.pi * features["hour_of_day"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour_of_day"] / 24)
    features["week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

    for lag in lags:
        lagged = weekly_seasonal_predictions(features, lag_hours=lag, timezone=timezone)[
            ["pickup_hour", "pickup_zone_id", "prediction"]
        ].rename(columns={"prediction": f"lag_{lag}h"})
        features = features.merge(
            lagged,
            on=["pickup_hour", "pickup_zone_id"],
            how="left",
            validate="1:1",
        )

    grouped = features.groupby("pickup_zone_id", sort=False)["pickup_count"]
    for window in rolling_windows:
        features[f"rolling_mean_{window}h"] = grouped.transform(
            lambda values, _window=window: (
                values.shift(1).rolling(_window, min_periods=_window).mean()
            )
        )
        features[f"rolling_std_{window}h"] = grouped.transform(
            lambda values, _window=window: (
                values.shift(1).rolling(_window, min_periods=_window).std()
            )
        )

    return features.sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(drop=True)
