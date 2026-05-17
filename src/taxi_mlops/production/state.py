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
    score_sha256: str | None = None


@dataclass(frozen=True)
class ModelDeployment:
    model_name: str
    model_version: str
    model_alias: str
    model_run_id: str
    deployed_at: str
    training_start: str
    training_end: str
    data_fingerprint: str
    reason: str


@dataclass(frozen=True)
class AlertIncident:
    incident_id: int
    alert_key: str
    status: str
    severity: str
    opened_at: str
    first_window_end: str
    last_observed_at: str
    last_window_end: str
    last_value: float
    threshold: float
    acknowledged_at: str | None
    acknowledged_by: str | None
    acknowledgement_note: str | None
    resolved_at: str | None


@dataclass(frozen=True)
class AlertTransition:
    transition_id: int
    incident_id: int
    occurred_at: str
    action: str
    from_status: str | None
    to_status: str
    from_severity: str | None
    to_severity: str
    details: dict[str, Any]


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
                CREATE TABLE IF NOT EXISTS model_deployments (
                    deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_alias TEXT NOT NULL,
                    model_run_id TEXT NOT NULL,
                    deployed_at TEXT NOT NULL,
                    training_start TEXT NOT NULL,
                    training_end TEXT NOT NULL,
                    data_fingerprint TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, model_version, model_alias)
                );
                CREATE TABLE IF NOT EXISTS alert_incidents (
                    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('open', 'acknowledged', 'resolved')),
                    severity TEXT NOT NULL CHECK (severity IN ('warning', 'severe')),
                    opened_at TEXT NOT NULL,
                    first_window_end TEXT NOT NULL,
                    last_observed_at TEXT NOT NULL,
                    last_window_end TEXT NOT NULL,
                    last_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    acknowledgement_note TEXT,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_alert_incidents_one_active
                    ON alert_incidents(alert_key)
                    WHERE status IN ('open', 'acknowledged');
                CREATE TABLE IF NOT EXISTS alert_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER NOT NULL REFERENCES alert_incidents(incident_id),
                    occurred_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    from_severity TEXT,
                    to_severity TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS alert_deliveries (
                    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER NOT NULL REFERENCES alert_incidents(incident_id),
                    action TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('delivered', 'failed')),
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            score_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(scores)").fetchall()
            }
            if "score_sha256" not in score_columns:
                connection.execute("ALTER TABLE scores ADD COLUMN score_sha256 TEXT")

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
                    , score_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(fields.values()),
            )
        return score

    def set_score_checksum(self, batch_id: str, checksum: str) -> ScoreRecord:
        """Backfill a checksum for score artifacts created before schema version 2."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scores
                SET score_sha256 = ?
                WHERE batch_id = ? AND score_sha256 IS NULL
                """,
                (checksum, batch_id),
            )
            if cursor.rowcount not in {0, 1}:
                raise RuntimeError("score checksum update affected an unexpected row count")
        score = self.score_for_batch(batch_id)
        if score is None:  # pragma: no cover - guarded by the update target
            raise RuntimeError(f"score disappeared during checksum update: {batch_id}")
        if score.score_sha256 != checksum:
            raise ValueError("persisted score checksum conflicts with the artifact")
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

    def latest_deployment(self, model_alias: str) -> ModelDeployment | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_name, model_version, model_alias, model_run_id,
                       deployed_at, training_start, training_end,
                       data_fingerprint, reason
                FROM model_deployments
                WHERE model_alias = ?
                ORDER BY deployed_at DESC, deployment_id DESC
                LIMIT 1
                """,
                (model_alias,),
            ).fetchone()
        return None if row is None else ModelDeployment(**dict(row))

    def record_deployment(self, deployment: ModelDeployment) -> ModelDeployment:
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT model_name, model_version, model_alias, model_run_id,
                       deployed_at, training_start, training_end,
                       data_fingerprint, reason
                FROM model_deployments
                WHERE model_name = ? AND model_version = ? AND model_alias = ?
                """,
                (
                    deployment.model_name,
                    deployment.model_version,
                    deployment.model_alias,
                ),
            ).fetchone()
            if existing is not None:
                current = ModelDeployment(**dict(existing))
                if current != deployment:
                    raise ValueError("registered deployment has conflicting lineage")
                return current
            connection.execute(
                """
                INSERT INTO model_deployments(
                    model_name, model_version, model_alias, model_run_id,
                    deployed_at, training_start, training_end,
                    data_fingerprint, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(asdict(deployment).values()),
            )
        return deployment

    def score_history(self) -> list[dict[str, Any]]:
        """Return scored batches with their immutable prediction lineage."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    p.batch_id, p.target_hour, p.generated_at,
                    p.model_name, p.model_version, p.model_alias, p.model_run_id,
                    p.feature_fingerprint, p.row_count,
                    s.scored_at, s.scored_path, s.observations,
                    s.actual_pickups, s.predicted_pickups, s.mae, s.wape,
                    s.mean_bias, s.prediction_coverage, s.score_sha256
                FROM prediction_batches AS p
                INNER JOIN scores AS s USING (batch_id)
                ORDER BY p.target_hour, p.batch_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def prediction_history(self) -> list[dict[str, Any]]:
        """Return every registered prediction batch, including unscored batches."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT batch_id, target_hour, generated_at, model_name,
                       model_version, model_alias, model_run_id, model_uri,
                       feature_fingerprint, prediction_path, prediction_sha256,
                       row_count
                FROM prediction_batches
                ORDER BY target_hour, batch_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def backup(self, destination: str | Path) -> Path:
        """Create a transactionally consistent online SQLite backup."""

        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        source = self._connect()
        target = sqlite3.connect(temporary)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
            source.close()
        try:
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def event_history(self) -> list[dict[str, Any]]:
        """Return the append-only pipeline audit trail with decoded details."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, occurred_at, stage, status, batch_id, details_json
                FROM pipeline_events
                ORDER BY event_id
                """
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            events.append(event)
        return events

    def deployment_history(self) -> list[dict[str, Any]]:
        """Return model deployment history in control-plane order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT deployment_id, model_name, model_version, model_alias,
                       model_run_id, deployed_at, training_start, training_end,
                       data_fingerprint, reason
                FROM model_deployments
                ORDER BY deployed_at, deployment_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def watermark_history(self) -> list[dict[str, Any]]:
        """Return current stage checkpoints for operational monitoring."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT stage, value, input_fingerprint, updated_at
                FROM watermarks
                ORDER BY stage
                """
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _alert_from_row(row: sqlite3.Row) -> AlertIncident:
        values = dict(row)
        values.pop("created_at", None)
        return AlertIncident(**values)

    @staticmethod
    def _record_alert_transition(
        connection: sqlite3.Connection,
        *,
        incident_id: int,
        occurred_at: str,
        action: str,
        from_status: str | None,
        to_status: str,
        from_severity: str | None,
        to_severity: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO alert_transitions(
                incident_id, occurred_at, action, from_status, to_status,
                from_severity, to_severity, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                occurred_at,
                action,
                from_status,
                to_status,
                from_severity,
                to_severity,
                json.dumps(details or {}, sort_keys=True),
            ),
        )

    def active_alert(self, alert_key: str) -> AlertIncident | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM alert_incidents
                WHERE alert_key = ? AND status IN ('open', 'acknowledged')
                ORDER BY incident_id DESC LIMIT 1
                """,
                (alert_key,),
            ).fetchone()
        return None if row is None else self._alert_from_row(row)

    def latest_alert(self, alert_key: str) -> AlertIncident | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM alert_incidents
                WHERE alert_key = ?
                ORDER BY incident_id DESC LIMIT 1
                """,
                (alert_key,),
            ).fetchone()
        return None if row is None else self._alert_from_row(row)

    def alert_incident(self, incident_id: int) -> AlertIncident | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return None if row is None else self._alert_from_row(row)

    def open_alert(
        self,
        *,
        alert_key: str,
        severity: str,
        observed_at: str,
        window_end: str,
        value: float,
        threshold: float,
    ) -> tuple[AlertIncident, bool]:
        """Open one incident atomically, returning an active duplicate when present."""

        if severity not in {"warning", "severe"}:
            raise ValueError(f"unsupported alert severity: {severity}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM alert_incidents
                WHERE alert_key = ? AND status IN ('open', 'acknowledged')
                ORDER BY incident_id DESC LIMIT 1
                """,
                (alert_key,),
            ).fetchone()
            if existing is not None:
                return self._alert_from_row(existing), False
            cursor = connection.execute(
                """
                INSERT INTO alert_incidents(
                    alert_key, status, severity, opened_at, first_window_end,
                    last_observed_at, last_window_end, last_value, threshold
                ) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_key,
                    severity,
                    observed_at,
                    window_end,
                    observed_at,
                    window_end,
                    value,
                    threshold,
                ),
            )
            incident_id = int(cursor.lastrowid)
            self._record_alert_transition(
                connection,
                incident_id=incident_id,
                occurred_at=observed_at,
                action="opened",
                from_status=None,
                to_status="open",
                from_severity=None,
                to_severity=severity,
                details={"value": value, "threshold": threshold, "window_end": window_end},
            )
            row = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return self._alert_from_row(row), True

    def observe_alert(
        self,
        incident_id: int,
        *,
        observed_at: str,
        window_end: str,
        value: float,
        threshold: float,
    ) -> AlertIncident:
        """Refresh an incident without adding a duplicate transition."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_incidents
                SET last_observed_at = ?, last_window_end = ?, last_value = ?, threshold = ?
                WHERE incident_id = ? AND status IN ('open', 'acknowledged')
                """,
                (observed_at, window_end, value, threshold, incident_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"alert incident is not active: {incident_id}")
            row = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return self._alert_from_row(row)

    def escalate_alert(
        self,
        incident_id: int,
        *,
        observed_at: str,
        window_end: str,
        value: float,
        threshold: float,
    ) -> AlertIncident:
        """Escalate warning to severe and re-open an acknowledged incident."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if row is None or row["status"] == "resolved":
                raise ValueError(f"alert incident is not active: {incident_id}")
            if row["severity"] != "warning":
                raise ValueError(f"alert incident is already severe: {incident_id}")
            previous_status = str(row["status"])
            connection.execute(
                """
                UPDATE alert_incidents
                SET status = 'open', severity = 'severe', last_observed_at = ?,
                    last_window_end = ?, last_value = ?, threshold = ?,
                    acknowledged_at = NULL, acknowledged_by = NULL,
                    acknowledgement_note = NULL
                WHERE incident_id = ?
                """,
                (observed_at, window_end, value, threshold, incident_id),
            )
            self._record_alert_transition(
                connection,
                incident_id=incident_id,
                occurred_at=observed_at,
                action="escalated",
                from_status=previous_status,
                to_status="open",
                from_severity="warning",
                to_severity="severe",
                details={"value": value, "threshold": threshold, "window_end": window_end},
            )
            updated = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return self._alert_from_row(updated)

    def acknowledge_alert(
        self,
        incident_id: int,
        *,
        acknowledged_at: str,
        actor: str,
        note: str | None = None,
    ) -> tuple[AlertIncident, bool]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown alert incident: {incident_id}")
            if row["status"] == "resolved":
                raise ValueError(f"cannot acknowledge resolved alert incident: {incident_id}")
            if row["status"] == "acknowledged":
                return self._alert_from_row(row), False
            connection.execute(
                """
                UPDATE alert_incidents
                SET status = 'acknowledged', acknowledged_at = ?,
                    acknowledged_by = ?, acknowledgement_note = ?
                WHERE incident_id = ?
                """,
                (acknowledged_at, actor, note, incident_id),
            )
            self._record_alert_transition(
                connection,
                incident_id=incident_id,
                occurred_at=acknowledged_at,
                action="acknowledged",
                from_status="open",
                to_status="acknowledged",
                from_severity=str(row["severity"]),
                to_severity=str(row["severity"]),
                details={"actor": actor, "note": note},
            )
            updated = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return self._alert_from_row(updated), True

    def resolve_alert(
        self,
        incident_id: int,
        *,
        resolved_at: str,
        window_end: str,
        value: float,
        threshold: float,
    ) -> AlertIncident:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if row is None or row["status"] == "resolved":
                raise ValueError(f"alert incident is not active: {incident_id}")
            previous_status = str(row["status"])
            severity = str(row["severity"])
            connection.execute(
                """
                UPDATE alert_incidents
                SET status = 'resolved', resolved_at = ?, last_observed_at = ?,
                    last_window_end = ?, last_value = ?, threshold = ?
                WHERE incident_id = ?
                """,
                (resolved_at, resolved_at, window_end, value, threshold, incident_id),
            )
            self._record_alert_transition(
                connection,
                incident_id=incident_id,
                occurred_at=resolved_at,
                action="resolved",
                from_status=previous_status,
                to_status="resolved",
                from_severity=severity,
                to_severity=severity,
                details={"value": value, "threshold": threshold, "window_end": window_end},
            )
            updated = connection.execute(
                "SELECT * FROM alert_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return self._alert_from_row(updated)

    def alert_history(self, *, active_only: bool = False) -> list[AlertIncident]:
        where = "WHERE status IN ('open', 'acknowledged')" if active_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM alert_incidents {where} ORDER BY incident_id DESC"
            ).fetchall()
        return [self._alert_from_row(row) for row in rows]

    def alert_transition_history(self, incident_id: int | None = None) -> list[AlertTransition]:
        query = "SELECT * FROM alert_transitions"
        parameters: tuple[object, ...] = ()
        if incident_id is not None:
            query += " WHERE incident_id = ?"
            parameters = (incident_id,)
        query += " ORDER BY transition_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        transitions = []
        for row in rows:
            values = dict(row)
            values.pop("created_at", None)
            values["details"] = json.loads(values.pop("details_json"))
            transitions.append(AlertTransition(**values))
        return transitions

    def record_alert_delivery(
        self,
        *,
        incident_id: int,
        action: str,
        attempted_at: str,
        status: str,
        error: str | None = None,
    ) -> None:
        if status not in {"delivered", "failed"}:
            raise ValueError(f"unsupported alert delivery status: {status}")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alert_deliveries(
                    incident_id, action, attempted_at, status, error
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (incident_id, action, attempted_at, status, error),
            )

    def alert_delivery_history(self, incident_id: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT delivery_id, incident_id, action, attempted_at, status, error
            FROM alert_deliveries
        """
        parameters: tuple[object, ...] = ()
        if incident_id is not None:
            query += " WHERE incident_id = ?"
            parameters = (incident_id,)
        query += " ORDER BY delivery_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]
