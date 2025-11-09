"""Period profiles and similarity measures for temporal-regime analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_monthly_regimes(
    data: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return monthly volume, normalized weekly profiles, and profile similarity."""

    required = {"pickup_hour", "pickup_zone_id", "pickup_count"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {', '.join(sorted(missing))}")

    city = data[["pickup_hour", "pickup_count"]].copy()
    city["pickup_hour"] = pd.to_datetime(city["pickup_hour"], utc=True, errors="raise")
    city = city.groupby("pickup_hour", as_index=False)["pickup_count"].sum()
    city = city.sort_values("pickup_hour")
    local = city["pickup_hour"].dt.tz_convert(timezone)
    city["month"] = local.dt.strftime("%Y-%m")
    city["hour_of_week"] = local.dt.dayofweek * 24 + local.dt.hour

    monthly = (
        city.groupby("month", as_index=False)
        .agg(
            total_pickups=("pickup_count", "sum"),
            mean_hourly_pickups=("pickup_count", "mean"),
            std_hourly_pickups=("pickup_count", "std"),
            observed_hours=("pickup_count", "size"),
        )
        .sort_values("month")
    )
    monthly["relative_mean_demand"] = (
        monthly["mean_hourly_pickups"] / monthly["mean_hourly_pickups"].mean()
    )
    monthly["month_over_month_mean_change"] = monthly["mean_hourly_pickups"].pct_change()

    profiles = (
        city.groupby(["month", "hour_of_week"], as_index=False)
        .agg(mean_pickups=("pickup_count", "mean"), observations=("pickup_count", "size"))
        .sort_values(["month", "hour_of_week"])
    )
    profile_means = profiles.groupby("month")["mean_pickups"].transform("mean")
    profiles["normalized_demand"] = np.divide(
        profiles["mean_pickups"],
        profile_means,
        out=np.zeros(len(profiles), dtype=float),
        where=profile_means.to_numpy() != 0,
    )

    wide = profiles.pivot(index="month", columns="hour_of_week", values="normalized_demand")
    similarity = wide.T.corr(method="pearson", min_periods=120)
    similarity.index.name = "month"
    similarity.columns.name = "comparison_month"
    return monthly, profiles, similarity
