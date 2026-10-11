"""Materialize fast, read-only per-zone history from registered scored artifacts."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from taxi_mlops.production.config import ProductionConfig


def zone_index_root(config: ProductionConfig) -> Path:
    return config.storage.monitoring_root / "zone_history_index"


def build_zone_history_index(config: ProductionConfig) -> dict[str, object]:
    """Build one small Parquet file per zone from registered score batches."""
    database = config.storage.state_db.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database, uri=True)) as connection:
        rows = connection.execute(
            "SELECT p.target_hour, s.scored_path FROM prediction_batches p "
            "JOIN scores s USING(batch_id) ORDER BY p.target_hour"
        ).fetchall()
    if not rows:
        raise ValueError("no registered scores to index")

    expected_ids = np.arange(1, config.expected_zones + 1)

    def read_artifact(row: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_parquet(
            row[1], columns=["pickup_zone_id", "pickup_count", "prediction"]
        ).sort_values("pickup_zone_id")
        if not np.array_equal(frame["pickup_zone_id"].to_numpy(), expected_ids):
            raise ValueError(f"incomplete taxi-zone grid in {row[1]}")
        return (
            frame["pickup_count"].to_numpy(dtype=np.int32),
            frame["prediction"].to_numpy(dtype=np.float64),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(read_artifact, rows))
    actual = np.stack([item[0] for item in values])
    forecast = np.stack([item[1] for item in values])
    hours = pd.to_datetime([row[0] for row in rows], utc=True)
    root = zone_index_root(config)
    root.mkdir(parents=True, exist_ok=True)
    for pos, zone_id in enumerate(expected_ids):
        path = root / f"zone_{zone_id:03d}.parquet"
        temporary = path.with_suffix(".tmp")
        pd.DataFrame(
            {
                "pickup_hour": hours,
                "pickup_count": actual[:, pos],
                "prediction": forecast[:, pos],
            }
        ).to_parquet(temporary, index=False)
        temporary.replace(path)
    manifest = {
        "first_scored_hour": rows[0][0],
        "last_scored_hour": rows[-1][0],
        "scored_hours": len(rows),
        "zones": config.expected_zones,
    }
    manifest_path = root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def indexed_zone_values(
    config: ProductionConfig,
    *,
    zone_id: int,
    start: pd.Timestamp,
    read_start: pd.Timestamp,
    end: pd.Timestamp,
    expected_rows: int,
) -> list[tuple[pd.Timestamp, float, float]] | None:
    """Return cached values only when the index covers the requested window."""
    root = zone_index_root(config)
    manifest_path = root / "manifest.json"
    zone_path = root / f"zone_{zone_id:03d}.parquet"
    if not manifest_path.is_file() or not zone_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = pd.Timestamp(manifest["first_scored_hour"])
    last = pd.Timestamp(manifest["last_scored_hour"])
    if first > start or last < end or manifest["zones"] != config.expected_zones:
        return None
    frame = pd.read_parquet(zone_path)
    frame = frame.loc[frame["pickup_hour"].between(read_start, end)]
    if len(frame) != expected_rows:
        return None
    return [
        (row.pickup_hour, float(row.pickup_count), float(row.prediction))
        for row in frame.itertuples(index=False)
    ]
