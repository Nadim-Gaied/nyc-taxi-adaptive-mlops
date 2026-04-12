"""Summaries that connect demand behavior to later modeling decisions."""

from __future__ import annotations

import math

import pandas as pd


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _validate_canonical_data(data: pd.DataFrame) -> pd.DataFrame:
    required = {"pickup_hour", "pickup_zone_id", "pickup_count"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {', '.join(sorted(missing))}")

    clean = data[list(required)].copy()
    clean["pickup_hour"] = pd.to_datetime(clean["pickup_hour"], utc=True, errors="raise")
    if clean.duplicated(["pickup_hour", "pickup_zone_id"]).any():
        raise ValueError("canonical data contains duplicate zone-hour keys")
    if clean["pickup_count"].isna().any() or clean["pickup_count"].lt(0).any():
        raise ValueError("pickup_count must be complete and non-negative")
    return clean.sort_values(["pickup_hour", "pickup_zone_id"])


def characterize_dataset(
    data: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Return a JSON-ready summary, hour-of-week profile, and zone summary."""

    clean = _validate_canonical_data(data)
    zones = int(clean["pickup_zone_id"].nunique())
    hours = int(clean["pickup_hour"].nunique())
    expected_rows = zones * hours
    if len(clean) != expected_rows:
        raise ValueError(
            f"canonical grid is incomplete: found {len(clean)} rows, expected {expected_rows}"
        )

    city = clean.groupby("pickup_hour", as_index=False)["pickup_count"].sum()
    local_time = city["pickup_hour"].dt.tz_convert(timezone)
    city["day_of_week"] = local_time.dt.dayofweek
    city["hour_of_day"] = local_time.dt.hour
    city["hour_of_week"] = city["day_of_week"] * 24 + city["hour_of_day"]

    autocorrelation: dict[str, float | None] = {}
    for lag in (1, 24, 168):
        value = city["pickup_count"].autocorr(lag=lag) if len(city) > lag else float("nan")
        autocorrelation[f"lag_{lag}h"] = _finite_or_none(float(value))

    hour_of_week = (
        city.groupby(["hour_of_week", "day_of_week", "hour_of_day"], as_index=False)
        .agg(
            mean_pickups=("pickup_count", "mean"),
            std_pickups=("pickup_count", "std"),
            observations=("pickup_count", "size"),
        )
        .sort_values("hour_of_week")
    )
    hour_of_week["std_pickups"] = hour_of_week["std_pickups"].fillna(0.0)

    zone_summary = (
        clean.groupby("pickup_zone_id", as_index=False)
        .agg(
            total_pickups=("pickup_count", "sum"),
            mean_hourly_pickups=("pickup_count", "mean"),
            median_hourly_pickups=("pickup_count", "median"),
            p95_hourly_pickups=("pickup_count", lambda values: values.quantile(0.95)),
            zero_demand_share=("pickup_count", lambda values: values.eq(0).mean()),
        )
        .sort_values("total_pickups", ascending=False)
        .reset_index(drop=True)
    )
    total_pickups = int(zone_summary["total_pickups"].sum())
    zone_summary["demand_share"] = (
        zone_summary["total_pickups"] / total_pickups if total_pickups else 0.0
    )
    zone_summary["cumulative_demand_share"] = zone_summary["demand_share"].cumsum()
    zone_summary["volume_rank"] = zone_summary.index + 1

    top_zones = [
        {
            "pickup_zone_id": int(row.pickup_zone_id),
            "total_pickups": int(row.total_pickups),
            "demand_share": float(row.demand_share),
        }
        for row in zone_summary.head(10).itertuples(index=False)
    ]
    city_counts = city["pickup_count"]
    summary: dict[str, object] = {
        "dataset": {
            "rows": len(clean),
            "zones": zones,
            "hours": hours,
            "start_utc": str(clean["pickup_hour"].min()),
            "end_utc": str(clean["pickup_hour"].max()),
            "total_pickups": total_pickups,
            "zero_demand_zone_hour_share": float(clean["pickup_count"].eq(0).mean()),
        },
        "city_hourly_demand": {
            "mean": float(city_counts.mean()),
            "median": float(city_counts.median()),
            "standard_deviation": float(city_counts.std()),
            "minimum": int(city_counts.min()),
            "maximum": int(city_counts.max()),
            "p05": float(city_counts.quantile(0.05)),
            "p95": float(city_counts.quantile(0.95)),
        },
        "autocorrelation": autocorrelation,
        "spatial_concentration": {
            "top_10_zone_share": float(zone_summary.head(10)["demand_share"].sum()),
            "top_25_zone_share": float(zone_summary.head(25)["demand_share"].sum()),
            "zones_for_80_percent_of_demand": int(
                (zone_summary["cumulative_demand_share"] < 0.8).sum() + 1
            ),
            "top_zones": top_zones,
        },
    }
    return summary, hour_of_week, zone_summary
