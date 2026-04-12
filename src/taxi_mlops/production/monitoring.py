"""Reproducible forecast-quality and operational monitoring artifacts."""

from __future__ import annotations

import base64
import html
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taxi_mlops.evaluation import (
    assign_demand_tiers,
    tier_regression_metrics,
    zone_regression_metrics,
)
from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.state import ProductionStateStore
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource


@dataclass(frozen=True)
class MonitoringReport:
    generated_at: str
    as_of: str
    forecast_status: str
    operational_status: str
    scored_hours: int
    alert_window_complete: bool
    alert_window_wape: float | None
    dashboard_path: str
    snapshot_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _aggregate_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    observations = int(frame["observations"].sum())
    actual = float(frame["actual_pickups"].sum())
    predicted = float(frame["predicted_pickups"].sum())
    absolute_error = float(frame["absolute_error_total"].sum())
    return {
        "hours": int(frame["target_hour"].nunique()),
        "observations": observations,
        "actual_pickups": actual,
        "predicted_pickups": predicted,
        "mae": absolute_error / observations if observations else None,
        "wape": absolute_error / actual if actual else None,
        "mean_bias": (predicted - actual) / observations if observations else None,
        "prediction_coverage": (
            float(np.average(frame["prediction_coverage"], weights=frame["observations"]))
            if observations
            else None
        ),
    }


def _score_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "target_hour",
                "model_version",
                "row_count",
                "scored_path",
                "observations",
                "actual_pickups",
                "predicted_pickups",
                "mae",
                "wape",
                "mean_bias",
                "prediction_coverage",
                "absolute_error_total",
            ]
        )
    frame = pd.DataFrame.from_records(records)
    frame["target_hour"] = pd.to_datetime(frame["target_hour"], utc=True, errors="raise")
    frame["scored_at"] = pd.to_datetime(frame["scored_at"], utc=True, errors="raise")
    frame["absolute_error_total"] = frame["mae"] * frame["observations"]
    return frame.sort_values(["target_hour", "batch_id"]).reset_index(drop=True)


def _weekly_metrics(hourly: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "week_start_local",
                "hours",
                "observations",
                "actual_pickups",
                "predicted_pickups",
                "mae",
                "wape",
                "mean_bias",
                "prediction_coverage",
            ]
        )
    local = hourly["target_hour"].dt.tz_convert(timezone)
    local_midnight = local.dt.normalize()
    hourly = hourly.assign(
        week_start_local=(local_midnight - pd.to_timedelta(local.dt.weekday, unit="D")).dt.date
    )
    rows = []
    for week_start, group in hourly.groupby("week_start_local", sort=True):
        rows.append({"week_start_local": str(week_start), **_aggregate_metrics(group)})
    return pd.DataFrame.from_records(rows)


def _operational_summary(events: list[dict[str, Any]]) -> tuple[str, pd.DataFrame]:
    columns = ["stage", "status", "event_count", "mean_duration_seconds"]
    if not events:
        return "no_data", pd.DataFrame(columns=columns)
    rows = []
    for event in events:
        details = event.get("details", {})
        duration = details.get("pipeline_seconds", details.get("training_seconds"))
        rows.append(
            {
                "event_id": event["event_id"],
                "stage": event["stage"],
                "status": event["status"],
                "duration_seconds": duration,
            }
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["stage", "status"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            mean_duration_seconds=("duration_seconds", "mean"),
        )
        .sort_values(["stage", "status"])
        .reset_index(drop=True)
    )
    last_by_stage = frame.sort_values("event_id").groupby("stage").tail(1)
    statuses = set(last_by_stage["status"])
    status = "severe" if "failed" in statuses else "warning" if "started" in statuses else "healthy"
    return status, summary[columns]


def _plot_weekly(weekly: pd.DataFrame, warning: float, severe: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    if weekly.empty:
        axis.text(0.5, 0.5, "No scored weeks", ha="center", va="center")
        axis.set_axis_off()
    else:
        x = pd.to_datetime(weekly["week_start_local"])
        axis.plot(x, weekly["wape"] * 100, marker="o", label="Observed WAPE")
        axis.axhline(warning * 100, color="#e69f00", linestyle="--", label="Warning")
        axis.axhline(severe * 100, color="#d55e00", linestyle="--", label="Severe")
        axis.set_ylabel("WAPE (%)")
        axis.set_xlabel("Local week starting Monday")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_zones(zones: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    eligible = zones.dropna(subset=["wape"]).sort_values(
        ["actual_pickups", "pickup_zone_id"], ascending=[False, True]
    )
    if eligible.empty:
        axis.text(0.5, 0.5, "No scored zone data", ha="center", va="center")
        axis.set_axis_off()
    else:
        x = np.arange(len(eligible))
        axis.bar(x, eligible["wape"] * 100, color="#0072b2", width=0.9)
        weighted_wape = zones["absolute_error_total"].sum() / zones["actual_pickups"].sum()
        axis.axhline(
            weighted_wape * 100,
            color="#d55e00",
            linestyle="--",
            label=f"Overall WAPE ({weighted_wape:.1%})",
        )
        axis.set_xlabel("Taxi zones ranked by observed pickup volume")
        axis.set_ylabel("Zone WAPE (%)")
        axis.set_xticks([])
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_operations(operations: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    if operations.empty:
        axis.text(0.5, 0.5, "No pipeline events", ha="center", va="center")
        axis.set_axis_off()
    else:
        pivot = operations.pivot(index="stage", columns="status", values="event_count").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=axis, color=["#009e73", "#d55e00", "#56b4e9"])
        axis.set_ylabel("Event count")
        axis.set_xlabel("Pipeline stage")
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _embedded_png(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _table_html(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    visible = frame if columns is None else frame[[column for column in columns if column in frame]]
    return visible.to_html(
        index=False, border=0, classes="metrics", na_rep="—", float_format="%.4f"
    )


class ProductionMonitoringReporter:
    """Build deterministic monitoring snapshots from durable production state."""

    def __init__(
        self,
        *,
        config: ProductionConfig,
        state_store: ProductionStateStore,
        zone_tiers: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.state = state_store
        self.zone_tiers = zone_tiers

    def _resolve_zone_tiers(self) -> pd.DataFrame | None:
        if self.zone_tiers is not None:
            return self.zone_tiers.copy()
        deployment = self.state.latest_deployment(self.config.tracking.production_alias)
        if deployment is None:
            return None
        cache = self.config.storage.monitoring_root / (
            f"zone_tiers_v{deployment.model_version}_{deployment.data_fingerprint[:12]}.parquet"
        )
        if cache.is_file():
            return pd.read_parquet(cache)
        start = canonical_utc(deployment.training_start, "training_start")
        end = canonical_utc(deployment.training_end, "training_end")
        source = MonthlyReplayDataSource(
            feature_root=self.config.storage.feature_root,
            start=start,
            end=end,
            timezone=self.config.timezone,
        )
        training = source.observations_as_of(end, start=start)
        tiers = assign_demand_tiers(training)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{uuid4().hex}.tmp")
        try:
            tiers.to_parquet(temporary, index=False)
            os.replace(temporary, cache)
        finally:
            if temporary.exists():
                temporary.unlink()
        return tiers

    def _scored_observations(self, hourly: pd.DataFrame, *, cutoff: pd.Timestamp) -> pd.DataFrame:
        if hourly.empty:
            return pd.DataFrame(
                columns=["pickup_hour", "pickup_zone_id", "pickup_count", "prediction"]
            )
        start = cutoff - pd.Timedelta(days=self.config.monitoring.zone_lookback_days)
        selected = hourly.loc[hourly["target_hour"].between(start, cutoff, inclusive="left")]
        frames = []
        for row in selected.itertuples(index=False):
            path = Path(row.scored_path)
            if not path.is_file():
                raise FileNotFoundError(f"registered score artifact is missing: {path}")
            scored = pd.read_parquet(path)
            if "batch_id" in scored and set(scored["batch_id"].astype(str)) != {row.batch_id}:
                raise ValueError(f"score artifact has conflicting batch lineage: {path}")
            frames.append(scored)
        if not frames:
            return pd.DataFrame(
                columns=["pickup_hour", "pickup_zone_id", "pickup_count", "prediction"]
            )
        observations = pd.concat(frames, ignore_index=True)
        observations["pickup_hour"] = pd.to_datetime(
            observations["pickup_hour"], utc=True, errors="raise"
        )
        return observations.sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(drop=True)

    def build(self, *, as_of: str | pd.Timestamp) -> MonitoringReport:
        cutoff = canonical_utc(as_of, "as_of")
        output = self.config.storage.monitoring_root
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        hourly = _score_frame(self.state.score_history())
        hourly = hourly.loc[hourly["target_hour"] < cutoff].copy()
        weekly = _weekly_metrics(hourly, self.config.monitoring.reporting_timezone)
        operational_status, operations = _operational_summary(self.state.event_history())

        alert_hours = self.config.monitoring.alert_window_hours
        if hourly.empty:
            alert_frame = hourly
            complete = False
        else:
            latest = hourly["target_hour"].max()
            expected = pd.date_range(
                latest - pd.Timedelta(hours=alert_hours - 1), latest, freq="h", tz="UTC"
            )
            alert_frame = hourly.loc[hourly["target_hour"].isin(expected)]
            complete = bool(
                set(expected) == set(alert_frame["target_hour"])
                and len(alert_frame) == alert_hours
                and alert_frame["row_count"].eq(self.config.expected_zones).all()
                and alert_frame["observations"].eq(self.config.expected_zones).all()
                and alert_frame["prediction_coverage"].eq(1.0).all()
            )
        alert_metrics = (
            _aggregate_metrics(alert_frame)
            if not alert_frame.empty
            else {
                "hours": 0,
                "observations": 0,
                "actual_pickups": 0.0,
                "predicted_pickups": 0.0,
                "mae": None,
                "wape": None,
                "mean_bias": None,
                "prediction_coverage": None,
            }
        )
        alert_wape = alert_metrics["wape"]
        if not complete or alert_wape is None:
            forecast_status = "insufficient_data"
        elif alert_wape >= self.config.monitoring.weekly_wape_severe:
            forecast_status = "severe"
        elif alert_wape >= self.config.monitoring.weekly_wape_warning:
            forecast_status = "warning"
        else:
            forecast_status = "healthy"

        observations = self._scored_observations(hourly, cutoff=cutoff)
        if observations.empty:
            zones = pd.DataFrame(
                columns=[
                    "pickup_zone_id",
                    "observations",
                    "actual_pickups",
                    "predicted_pickups",
                    "mean_actual",
                    "mae",
                    "absolute_error_total",
                    "mean_bias",
                    "wape",
                    "prediction_coverage",
                ]
            )
            tiers = pd.DataFrame()
        else:
            zones = zone_regression_metrics(observations)
            zone_tiers = self._resolve_zone_tiers()
            tiers = (
                tier_regression_metrics(observations, zone_tiers)
                if zone_tiers is not None
                else pd.DataFrame()
            )

        hourly_export = hourly.copy()
        if not hourly_export.empty:
            hourly_export["target_hour"] = hourly_export["target_hour"].map(pd.Timestamp.isoformat)
            hourly_export["scored_at"] = hourly_export["scored_at"].map(pd.Timestamp.isoformat)
        _atomic_csv(hourly_export, output / "hourly_metrics.csv")
        _atomic_csv(weekly, output / "weekly_metrics.csv")
        _atomic_csv(zones, output / "zone_metrics.csv")
        _atomic_csv(tiers, output / "tier_metrics.csv")
        _atomic_csv(operations, output / "operational_metrics.csv")

        weekly_figure = figures / "weekly_wape.png"
        zone_figure = figures / "zone_wape.png"
        operations_figure = figures / "pipeline_events.png"
        _plot_weekly(
            weekly,
            self.config.monitoring.weekly_wape_warning,
            self.config.monitoring.weekly_wape_severe,
            weekly_figure,
        )
        _plot_zones(zones, zone_figure)
        _plot_operations(operations, operations_figure)

        latest_score = None if hourly.empty else hourly.iloc[-1]
        snapshot = {
            "schema_version": 1,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "as_of": cutoff.isoformat(),
            "forecast_status": forecast_status,
            "operational_status": operational_status,
            "monitoring_policy": {
                "alert_window_hours": alert_hours,
                "weekly_wape_warning": self.config.monitoring.weekly_wape_warning,
                "weekly_wape_severe": self.config.monitoring.weekly_wape_severe,
                "drift_can_trigger_retraining_alone": (
                    self.config.monitoring.drift_can_trigger_retraining_alone
                ),
            },
            "alert_window": {"complete": complete, **alert_metrics},
            "latest_score": None
            if latest_score is None
            else {
                "target_hour": latest_score["target_hour"].isoformat(),
                "model_version": str(latest_score["model_version"]),
                "wape": latest_score["wape"],
                "mae": latest_score["mae"],
                "prediction_coverage": latest_score["prediction_coverage"],
            },
            "counts": {
                "scored_hours": len(hourly),
                "scored_zones": int(len(zones)),
                "deployments": len(self.state.deployment_history()),
            },
            "watermarks": self.state.watermark_history(),
            "artifacts": {
                "hourly_metrics": "hourly_metrics.csv",
                "weekly_metrics": "weekly_metrics.csv",
                "zone_metrics": "zone_metrics.csv",
                "tier_metrics": "tier_metrics.csv",
                "operational_metrics": "operational_metrics.csv",
                "dashboard": "dashboard.html",
            },
        }
        snapshot = _json_safe(snapshot)
        snapshot_path = output / "snapshot.json"
        _atomic_text(
            snapshot_path,
            json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False),
        )

        status_class = html.escape(forecast_status)
        operational_class = html.escape(operational_status)
        escaped_forecast = html.escape(forecast_status)
        escaped_operational = html.escape(operational_status)
        alert_display = "&mdash;" if alert_wape is None else f"{alert_wape:.2%}"
        zone_table = _table_html(
            zones.head(20),
            ["pickup_zone_id", "actual_pickups", "mae", "wape", "mean_bias"],
        )
        dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>NYC Taxi Production Monitoring</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;
padding:0 1rem;color:#17202a}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem}}
.card{{padding:1rem;border:1px solid #d5d8dc;border-radius:8px;background:#fafafa}}
.healthy{{color:#087f5b}} .warning{{color:#b36b00}} .severe{{color:#c92a2a}}
.insufficient_data,.no_data{{color:#5f6b7a}} img{{max-width:100%;height:auto}}
table.metrics{{border-collapse:collapse;width:100%;font-size:.9rem}}
.metrics th,.metrics td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:right}}
.metrics th:first-child,.metrics td:first-child{{text-align:left}}
code{{background:#eee;padding:.1rem .3rem}}
</style></head><body>
<h1>NYC Taxi production monitoring</h1>
<p>Snapshot as of <code>{html.escape(cutoff.isoformat())}</code>. Alerts require a complete
{alert_hours}-hour window; drift alone cannot trigger retraining.</p>
<div class="cards">
<div class="card"><strong>Forecast status</strong><br>
<span class="{status_class}">{escaped_forecast}</span></div>
<div class="card"><strong>Operational status</strong><br>
<span class="{operational_class}">{escaped_operational}</span></div>
<div class="card"><strong>Scored hours</strong><br>{len(hourly)}</div>
<div class="card"><strong>Alert-window WAPE</strong><br>{alert_display}</div>
</div>
<h2>Forecast quality</h2><img alt="Weekly WAPE" src="{_embedded_png(weekly_figure)}">
{_table_html(weekly.tail(12))}
<h2>Zone tail</h2><p>Zones are ranked by pickup volume so WAPE is not confused
with scale-driven MAE.</p>
<img alt="Zone WAPE" src="{_embedded_png(zone_figure)}">
{zone_table}
<h2>Demand tiers</h2>{_table_html(tiers)}
<h2>Pipeline operations</h2>
<img alt="Pipeline event counts" src="{_embedded_png(operations_figure)}">
{_table_html(operations)}
</body></html>"""
        dashboard_path = output / "dashboard.html"
        _atomic_text(dashboard_path, dashboard)

        return MonitoringReport(
            generated_at=str(snapshot["generated_at"]),
            as_of=cutoff.isoformat(),
            forecast_status=forecast_status,
            operational_status=operational_status,
            scored_hours=len(hourly),
            alert_window_complete=complete,
            alert_window_wape=None if alert_wape is None else float(alert_wape),
            dashboard_path=str(dashboard_path.resolve()),
            snapshot_path=str(snapshot_path.resolve()),
        )
