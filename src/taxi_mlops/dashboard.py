"""Local read-only data feed for the taxi-zone dashboard."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from threading import Lock
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.dashboard_index import indexed_zone_values


def _hour(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"target hour has no UTC offset: {value}")
    return parsed.astimezone(UTC)


def _registered_rows(
    config: ProductionConfig, limit: int
) -> tuple[list[sqlite3.Row], sqlite3.Row | None]:
    database = config.storage.state_db.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        scored = connection.execute(
            """
            SELECT p.batch_id, p.target_hour, p.model_version, s.scored_path
            FROM prediction_batches AS p
            JOIN scores AS s USING (batch_id)
            ORDER BY p.target_hour DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        if not scored:
            raise ValueError("no scored production batches are registered")
        scored.reverse()
        next_hour = (_hour(scored[-1]["target_hour"]) + timedelta(hours=1)).isoformat()
        pending = connection.execute(
            """
            SELECT p.batch_id, p.target_hour, p.model_version, p.prediction_path
            FROM prediction_batches AS p
            LEFT JOIN scores AS s USING (batch_id)
            WHERE p.target_hour = ? AND s.batch_id IS NULL
            """,
            (next_hour,),
        ).fetchone()
    for previous, current in zip(scored, scored[1:], strict=False):
        if _hour(current["target_hour"]) - _hour(previous["target_hour"]) != timedelta(hours=1):
            raise ValueError("recent scored hours are not continuous")
    return scored, pending


def _version(scored: list[sqlite3.Row], pending: sqlite3.Row | None) -> str:
    identities = [row["batch_id"] for row in scored]
    if pending is not None:
        identities.append(pending["batch_id"])
    return hashlib.sha256("|".join(identities).encode()).hexdigest()[:16]


def _zone_values(
    path: str, *, target_hour: str, scored: bool, expected_zones: int
) -> tuple[list[float], list[int] | None]:
    columns = ["pickup_zone_id", "prediction"]
    if scored:
        columns.append("pickup_count")
    frame = pd.read_parquet(Path(path), columns=columns).sort_values("pickup_zone_id")
    zone_ids = frame["pickup_zone_id"].astype(int).tolist()
    if zone_ids != list(range(1, expected_zones + 1)):
        raise ValueError(f"artifact for {target_hour} has an incomplete or unordered zone grid")
    predictions = frame["prediction"].astype(float).tolist()
    if not all(pd.notna(value) and value >= 0 for value in predictions):
        raise ValueError(f"artifact for {target_hour} contains invalid predictions")
    actuals = frame["pickup_count"].astype(int).tolist() if scored else None
    return predictions, actuals


def _display_hour(target_hour: str, timezone_name: str) -> str:
    local = _hour(target_hour).astimezone(ZoneInfo(timezone_name))
    return f"{local:%b} {local.day}, {local.year} at {local:%H:%M %Z}"


def _decisions(hours: list[dict[str, object]], timezone_name: str) -> dict[str, object]:
    pending = hours[-1] if hours[-1]["a"] is None else None
    if pending is None:
        return {"status": "no_pending_forecast", "items": []}
    scored = hours[:-1]
    if not scored or _hour(pending["utc"]) - _hour(scored[-1]["utc"]) != timedelta(hours=1):
        return {"status": "stale_replay_labels", "items": []}
    target_clock = _hour(pending["utc"]).astimezone(ZoneInfo(timezone_name)).hour
    peers = [
        hour
        for hour in scored
        if _hour(hour["utc"]).astimezone(ZoneInfo(timezone_name)).hour == target_clock
    ]
    if len(peers) < 3 or len(scored) < 24:
        return {"status": "insufficient_history", "items": []}
    items = []
    for pos, forecast in enumerate(pending["p"]):
        baseline = median(hour["a"][pos] for hour in peers)
        if forecast < 2 * max(10, baseline) or forecast - baseline < 40:
            continue
        recent = scored[-24:]
        actual = sum(hour["a"][pos] for hour in recent)
        error = sum(abs(hour["a"][pos] - hour["p"][pos]) for hour in recent)
        wape = 100 * error / actual if actual else None
        reliable = actual >= 150 and wape is not None and wape < 40
        items.append(
            {
                "zone_id": pos + 1,
                "forecast": round(forecast, 1),
                "same_hour_baseline": round(baseline, 1),
                "uplift": round(forecast - baseline, 1),
                "ratio": round(forecast / max(10, baseline), 2),
                "recent_wape": round(wape, 1) if wape is not None else None,
                "status": "review_positioning" if reliable else "withheld_low_confidence",
                "reason": "forecast surge" if reliable else "low volume or high recent error",
            }
        )
    items.sort(key=lambda item: item["uplift"], reverse=True)
    return {
        "status": "ready" if items else "no_unusual_forecast_surge",
        "virtual_clock": pending["utc"],
        "source": "historical replay; labels released after each simulated hour",
        "items": items[:8],
    }


def _zone_history(
    config: ProductionConfig, *, zone_id: int, duration: str, end: str
) -> dict[str, object]:
    if not 1 <= zone_id <= config.expected_zones:
        raise ValueError("zone is outside the taxi-zone grid")
    if duration not in {"month", "year"}:
        raise ValueError("history duration must be month or year")
    end_hour = _hour(end)
    span = timedelta(days=30 if duration == "month" else 365)
    start = end_hour - span + timedelta(hours=1)
    # A week of earlier observations gives same-hour surge rules a causal baseline.
    read_start = start - timedelta(days=7)
    database = config.storage.state_db.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database, uri=True)) as connection:
        rows = connection.execute(
            "SELECT p.target_hour, s.scored_path FROM prediction_batches p "
            "JOIN scores s USING(batch_id) WHERE p.target_hour >= ? AND p.target_hour <= ? "
            "ORDER BY p.target_hour",
            (read_start.isoformat(), end_hour.isoformat()),
        ).fetchall()
    if not rows:
        return {
            "duration": duration,
            "sampling": "hour" if duration == "month" else "day",
            "samples": [],
            "scored_hours": 0,
            "expected_hours": int((end_hour - start) / timedelta(hours=1)) + 1,
        }

    def read_row(row: tuple[str, str]) -> tuple[datetime, float, float]:
        frame = pd.read_parquet(row[1], columns=["pickup_zone_id", "pickup_count", "prediction"])
        match = frame.loc[frame["pickup_zone_id"].eq(zone_id)]
        if len(match) != 1:
            raise ValueError(f"scored artifact has no unique zone {zone_id}: {row[1]}")
        return (
            _hour(row[0]),
            float(match["pickup_count"].iloc[0]),
            float(match["prediction"].iloc[0]),
        )

    indexed = indexed_zone_values(
        config,
        zone_id=zone_id,
        start=pd.Timestamp(start),
        read_start=pd.Timestamp(read_start),
        end=pd.Timestamp(end_hour),
        expected_rows=len(rows),
    )
    if indexed is None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(read_row, rows))
    else:
        values = indexed
    local_zone = ZoneInfo(config.timezone)
    prior: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=7))
    trailing: deque[tuple[float, float]] = deque(maxlen=24)
    bias_run = 0
    bias_direction = 0
    hourly = []
    for target, actual, forecast in values:
        local = target.astimezone(local_zone)
        peers = prior[local.hour]
        baseline = median(peers) if len(peers) >= 3 else None
        surge = baseline is not None and actual >= 2 * max(10, baseline) and actual - baseline >= 40
        peers.append(actual)
        trailing.append((actual, forecast))
        volume = sum(a for a, _ in trailing)
        error = sum(abs(a - p) for a, p in trailing)
        bias = 100 * (sum(p for _, p in trailing) - volume) / volume if volume else None
        wape = 100 * error / volume if len(trailing) == 24 and volume >= 50 else None
        direction = 1 if bias is not None and bias > 0 else -1
        if len(trailing) == 24 and volume >= 500 and abs(bias or 0) >= 15:
            bias_run = bias_run + 1 if direction == bias_direction else 1
            bias_direction = direction
        else:
            bias_run = 0
            bias_direction = 0
        if target < start:
            continue
        hourly.append(
            {
                "utc": target.isoformat(),
                "local": _display_hour(target.isoformat(), config.timezone),
                "actual": actual,
                "forecast": forecast,
                "absolute_error": abs(actual - forecast),
                "wape": wape,
                "bias": bias,
                "signal": {
                    "surge": surge,
                    "error": wape is not None and volume >= 150 and wape >= 40,
                    "bias": bias_run >= 6,
                },
            }
        )
    if duration == "month":
        samples = hourly
    else:
        days: dict[str, list[dict[str, object]]] = defaultdict(list)
        expected_by_day: dict[str, int] = defaultdict(int)
        for expected in pd.date_range(start, end_hour, freq="h"):
            expected_by_day[expected.tz_convert(local_zone).date().isoformat()] += 1
        for item in hourly:
            day = _hour(item["utc"]).astimezone(local_zone).date().isoformat()
            days[day].append(item)
        samples = []
        for day, group in days.items():
            actual = sum(item["actual"] for item in group)
            forecast = sum(item["forecast"] for item in group)
            error = sum(abs(item["actual"] - item["forecast"]) for item in group)
            samples.append(
                {
                    "utc": group[-1]["utc"],
                    "local": day,
                    "actual": actual,
                    "forecast": forecast,
                    "absolute_error": error,
                    "wape": 100 * error / actual if actual >= 50 else None,
                    "bias": 100 * (forecast - actual) / actual if actual else None,
                    "coverage_hours": len(group),
                    "expected_hours": expected_by_day[day],
                    "signal": {
                        key: any(item["signal"][key] for item in group)
                        for key in ("surge", "error", "bias")
                    },
                }
            )
    return {
        "duration": duration,
        "sampling": "hour" if duration == "month" else "day",
        "zone_id": zone_id,
        "start": start.isoformat(),
        "end": end_hour.isoformat(),
        "samples": samples,
        "scored_hours": len(hourly),
        "expected_hours": int((end_hour - start) / timedelta(hours=1)) + 1,
    }


class ReplayFeed:
    def __init__(self, config: ProductionConfig, *, scored_hours: int = 168) -> None:
        self.config = config
        self.scored_hours = scored_hours
        self._lock = Lock()
        self._cached: dict[str, object] | None = None

    def status(self) -> dict[str, object]:
        scored, pending = _registered_rows(self.config, self.scored_hours)
        return {
            "version": _version(scored, pending),
            "latest_scored_hour": scored[-1]["target_hour"],
            "next_forecast_hour": pending["target_hour"] if pending is not None else None,
        }

    def data(self) -> dict[str, object]:
        with self._lock:
            scored, pending = _registered_rows(self.config, self.scored_hours)
            version = _version(scored, pending)
            if self._cached is not None and self._cached["version"] == version:
                return self._cached
            hours = []
            for row in [*scored, *([pending] if pending is not None else [])]:
                is_scored = "scored_path" in row.keys()
                target_hour = row["target_hour"]
                path = row["scored_path"] if is_scored else row["prediction_path"]
                predictions, actuals = _zone_values(
                    path,
                    target_hour=target_hour,
                    scored=is_scored,
                    expected_zones=self.config.expected_zones,
                )
                hours.append(
                    {
                        "utc": target_hour,
                        "local": _display_hour(target_hour, self.config.timezone),
                        "a": actuals,
                        "p": predictions,
                        "pt": sum(predictions),
                        "ae": (
                            sum(
                                abs(actual - forecast)
                                for actual, forecast in zip(actuals, predictions, strict=True)
                            )
                            if actuals is not None
                            else None
                        ),
                    }
                )
            self._cached = {
                "version": version,
                "source": "registered production scores and predictions",
                "zone_ids": list(range(1, self.config.expected_zones + 1)),
                "latest_scored_hour": scored[-1]["target_hour"],
                "next_forecast_hour": pending["target_hour"] if pending is not None else None,
                "model_version": (pending or scored[-1])["model_version"],
                "hours": hours,
                "decisions": _decisions(hours, self.config.timezone),
            }
            return self._cached

    def history(self, *, zone_id: int, duration: str, end: str) -> dict[str, object]:
        return _zone_history(self.config, zone_id=zone_id, duration=duration, end=end)


def serve_dashboard(
    config: ProductionConfig,
    *,
    html_path: str | Path,
    replay_config: ProductionConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve the dashboard and its replay feed on the local machine."""

    feeds = {"archive": ReplayFeed(config)}
    if replay_config is not None:
        feeds["replay"] = ReplayFeed(replay_config)
    html_file = Path(html_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            route = parsed.path
            try:
                params = parse_qs(parsed.query)
                mode = params.get("feed", ["archive"])[0]
                if mode not in feeds:
                    raise ValueError(f"unknown dashboard feed: {mode}")
                feed = feeds[mode]
                if route in {"/", "/dashboard"}:
                    content = html_file.read_bytes()
                    mime = "text/html; charset=utf-8"
                elif route == "/api/feeds":
                    content = json.dumps({"feeds": list(feeds)}).encode("utf-8")
                    mime = "application/json; charset=utf-8"
                elif route == "/api/replay":
                    content = json.dumps(feed.data(), allow_nan=False).encode("utf-8")
                    mime = "application/json; charset=utf-8"
                elif route == "/api/status":
                    content = json.dumps(feed.status(), allow_nan=False).encode("utf-8")
                    mime = "application/json; charset=utf-8"
                elif route == "/api/history":
                    content = json.dumps(
                        feed.history(
                            zone_id=int(params["zone"][0]),
                            duration=params["duration"][0],
                            end=params["end"][0],
                        ),
                        allow_nan=False,
                    ).encode("utf-8")
                    mime = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError, KeyError, IndexError, sqlite3.Error) as error:
                content = json.dumps({"error": str(error)}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(f"Taxi dashboard: http://{host}:{port}/dashboard", flush=True)
        server.serve_forever()
