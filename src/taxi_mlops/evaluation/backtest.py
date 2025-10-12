"""Rolling chronological evaluation for leakage-safe seasonal baselines."""

from __future__ import annotations

import pandas as pd

from taxi_mlops.baselines import daily_weekly_blend_predictions, weekly_seasonal_predictions
from taxi_mlops.evaluation.metrics import regression_metrics


def _baseline_candidates(data: pd.DataFrame, timezone: str) -> list[pd.DataFrame]:
    return [
        weekly_seasonal_predictions(data, lag_hours=1, timezone=timezone),
        weekly_seasonal_predictions(data, lag_hours=24, timezone=timezone),
        weekly_seasonal_predictions(data, lag_hours=168, timezone=timezone),
        daily_weekly_blend_predictions(data, daily_weight=0.5, timezone=timezone),
    ]


def rolling_baseline_backtest(
    data: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp | None = None,
    timezone: str = "America/New_York",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate all baselines on common ISO calendar weeks."""

    if data.empty:
        raise ValueError("cannot backtest an empty dataset")
    timestamps = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
    if evaluation_start is None:
        evaluation_start_utc = timestamps.min() + pd.Timedelta(hours=168)
    else:
        evaluation_start_utc = pd.Timestamp(evaluation_start)
        if evaluation_start_utc.tzinfo is None:
            raise ValueError("evaluation_start must include an explicit UTC offset")
        evaluation_start_utc = evaluation_start_utc.tz_convert("UTC")

    weekly_rows: list[dict[str, object]] = []
    for candidate in _baseline_candidates(data, timezone):
        eligible = candidate.loc[candidate["pickup_hour"] >= evaluation_start_utc].copy()
        local_time = eligible["pickup_hour"].dt.tz_convert(timezone)
        eligible["evaluation_week"] = local_time.dt.strftime("%G-W%V")
        model_id = str(candidate["model_id"].iloc[0])
        for week, frame in eligible.groupby("evaluation_week", sort=True):
            metrics = regression_metrics(frame)
            weekly_rows.append(
                {
                    "model_id": model_id,
                    "evaluation_week": week,
                    "period_start_utc": str(frame["pickup_hour"].min()),
                    "period_end_utc": str(frame["pickup_hour"].max()),
                    **metrics,
                }
            )

    weekly = pd.DataFrame(weekly_rows).sort_values(["model_id", "evaluation_week"])
    model_summaries: list[dict[str, object]] = []
    for model_id, frame in weekly.groupby("model_id", sort=True):
        worst = frame.loc[frame["mae"].idxmax()]
        model_summaries.append(
            {
                "model_id": model_id,
                "evaluated_weeks": len(frame),
                "mean_weekly_mae": float(frame["mae"].mean()),
                "median_weekly_mae": float(frame["mae"].median()),
                "p90_weekly_mae": float(frame["mae"].quantile(0.90)),
                "worst_weekly_mae": float(worst["mae"]),
                "worst_week": str(worst["evaluation_week"]),
                "mean_weekly_wape": float(frame["wape"].mean()),
                "minimum_prediction_coverage": float(frame["prediction_coverage"].min()),
            }
        )
    model_summaries.sort(key=lambda item: float(item["mean_weekly_mae"]))
    summary = {
        "evaluation_start_utc": str(evaluation_start_utc),
        "evaluation_end_utc": str(timestamps.max()),
        "ranking_metric": "mean_weekly_mae",
        "models_ranked": model_summaries,
    }
    return weekly.reset_index(drop=True), summary

