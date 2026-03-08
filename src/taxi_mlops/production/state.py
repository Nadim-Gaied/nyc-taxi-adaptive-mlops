"""SQLite-backed production state with idempotent batch and score records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionBatch:
    batch_id: str
    target_hour: str
    generated_at: str
    model_name: str
    model_version: str
    model_alias: str
    model_run_id: str
    model_uri: str
    feature_fingerprint: str
    prediction_path: str
    prediction_sha256: str
    row_count: int


@dataclass(frozen=True)
class ScoreRecord:
    batch_id: str
    scored_at: str
    scored_path: str
    observations: int
    actual_pickups: float
    predicted_pickups: float
    mae: float
    wape: float | None
    mean_bias: float
    prediction_coverage: float


class ProductionStateStore:
    """Durable control-plane metadata; bulk predictions remain in Parquet."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS prediction_batches (
                    batch_id TEXT PRIMARY KEY,
                    target_hour TEXT NOT NULL UNIQUE,
                    generated_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_alias TEXT NOT NULL,
                    model_run_id TEXT NOT NULL,
                    model_uri TEXT NOT NULL,
                    feature_fingerprint TEXT NOT NULL,
                    prediction_path TEXT NOT NULL,
                    prediction_sha256 TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count > 0),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS scores (
                    batch_id TEXT PRIMARY KEY REFERENCES prediction_batches(batch_id),
                    scored_at TEXT NOT NULL,
                    scored_path TEXT NOT NULL,
                    observations INTEGER NOT NULL CHECK (observations > 0),
                    actual_pickups REAL NOT NULL,
                    predicted_pickups REAL NOT NULL,
                    mae REAL NOT NULL,
                    wape REAL,
                    mean_bias REAL NOT NULL,
                    prediction_coverage REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS pipeline_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    batch_id TEXT,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS watermarks (
                    stage TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def record_event(
        self,
        stage: str,
        status: str,
        *,
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pipeline_events(stage, status, batch_id, details_json)
                VALUES (?, ?, ?, ?)
                """,
                (stage, status, batch_id, json.dumps(details or {}, sort_keys=True)),
            )

    def prediction_for_hour(self, target_hour: str) -> PredictionBatch | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM prediction_batches WHERE target_hour = ?", (target_hour,)
            ).fetchone()
        if row is None:
            return None
        values = dict(row)
        values.pop("created_at")
        return PredictionBatch(**values)

    def record_prediction(self, batch: PredictionBatch) -> PredictionBatch:
        existing = self.prediction_for_hour(batch.target_hour)
        if existing is not None:
            if existing != batch:
                raise ValueError(
                    "target hour already has a prediction batch with different lineage"
                )
            return existing
        fields = asdict(batch)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO prediction_batches(
                    batch_id, target_hour, generated_at, model_name, model_version,
                    model_alias, model_run_id, model_uri, feature_fingerprint,
                    prediction_path, prediction_sha256, row_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(fields.values()),
            )
        return batch

    def score_for_batch(self, batch_id: str) -> ScoreRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scores WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return None
        values = dict(row)
        values.pop("created_at")
        return ScoreRecord(**values)

    def record_score(self, score: ScoreRecord) -> ScoreRecord:
        existing = self.score_for_batch(score.batch_id)
        if existing is not None:
            if existing != score:
                raise ValueError("prediction batch already has a different score record")
            return existing
        fields = asdict(score)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scores(
                    batch_id, scored_at, scored_path, observations, actual_pickups,
                    predicted_pickups, mae, wape, mean_bias, prediction_coverage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(fields.values()),
            )
        return score

    def update_watermark(self, stage: str, value: str, input_fingerprint: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO watermarks(stage, value, input_fingerprint)
                VALUES (?, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    value = excluded.value,
                    input_fingerprint = excluded.input_fingerprint,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (stage, value, input_fingerprint),
            )
