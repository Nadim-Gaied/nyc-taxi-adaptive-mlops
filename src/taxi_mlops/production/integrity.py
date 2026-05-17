"""Integrity auditing and recoverable control-plane checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from taxi_mlops.evaluation import regression_metrics
from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.state import ProductionStateStore
from taxi_mlops.tracking.fingerprints import sha256_file


@dataclass(frozen=True)
class IntegrityIssue:
    severity: str
    code: str
    message: str
    artifact: str | None = None
    batch_id: str | None = None


@dataclass(frozen=True)
class IntegrityReport:
    generated_at: str
    status: str
    prediction_batches: int
    score_records: int
    deployments: int
    issues: tuple[IntegrityIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _close(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


class ProductionIntegrityAuditor:
    """Prove control-plane metadata and immutable artifacts still agree."""

    def __init__(self, *, config: ProductionConfig, state_store: ProductionStateStore) -> None:
        self.config = config
        self.state = state_store

    def _prediction_issues(self, record: dict[str, Any]) -> list[IntegrityIssue]:
        issues: list[IntegrityIssue] = []
        batch_id = str(record["batch_id"])
        path = Path(record["prediction_path"])
        if not path.is_file():
            return [
                IntegrityIssue(
                    "error",
                    "prediction_missing",
                    "prediction artifact is missing",
                    str(path),
                    batch_id,
                )
            ]
        if sha256_file(path) != record["prediction_sha256"]:
            issues.append(
                IntegrityIssue(
                    "error",
                    "prediction_checksum_mismatch",
                    "prediction artifact no longer matches its registered checksum",
                    str(path),
                    batch_id,
                )
            )
            return issues
        try:
            frame = pd.read_parquet(path)
            if len(frame) != record["row_count"]:
                raise ValueError("row count differs from the control-plane record")
            if frame.duplicated(["pickup_hour", "pickup_zone_id"]).any():
                raise ValueError("duplicate zone-hour keys")
            for column, expected in {
                "batch_id": batch_id,
                "model_name": str(record["model_name"]),
                "model_version": str(record["model_version"]),
                "model_run_id": str(record["model_run_id"]),
                "feature_fingerprint": str(record["feature_fingerprint"]),
            }.items():
                if column not in frame or set(frame[column].astype(str)) != {expected}:
                    raise ValueError(f"{column} lineage differs from the control plane")
        except Exception as error:
            issues.append(
                IntegrityIssue(
                    "error",
                    "prediction_content_invalid",
                    str(error),
                    str(path),
                    batch_id,
                )
            )
        return issues

    def _score_issues(self, record: dict[str, Any]) -> list[IntegrityIssue]:
        issues: list[IntegrityIssue] = []
        batch_id = str(record["batch_id"])
        path = Path(record["scored_path"])
        if not path.is_file():
            return [
                IntegrityIssue(
                    "error", "score_missing", "score artifact is missing", str(path), batch_id
                )
            ]
        checksum = sha256_file(path)
        if record["score_sha256"] is None:
            issues.append(
                IntegrityIssue(
                    "warning",
                    "legacy_score_without_checksum",
                    "score predates checksum enforcement; rerun scoring to backfill it",
                    str(path),
                    batch_id,
                )
            )
        elif checksum != record["score_sha256"]:
            return [
                IntegrityIssue(
                    "error",
                    "score_checksum_mismatch",
                    "score artifact no longer matches its registered checksum",
                    str(path),
                    batch_id,
                )
            ]
        try:
            frame = pd.read_parquet(path)
            if "batch_id" not in frame or set(frame["batch_id"].astype(str)) != {batch_id}:
                raise ValueError("batch lineage differs from the control plane")
            metrics = regression_metrics(frame)
            comparisons = {
                "observations": (float(metrics["observations"]), float(record["observations"])),
                "actual_pickups": (float(frame["pickup_count"].sum()), record["actual_pickups"]),
                "predicted_pickups": (
                    float(frame["prediction"].sum()),
                    record["predicted_pickups"],
                ),
                "mae": (float(metrics["mae"]), record["mae"]),
                "wape": (metrics["wape"], record["wape"]),
                "mean_bias": (float(metrics["mean_bias"]), record["mean_bias"]),
                "prediction_coverage": (
                    float(metrics["prediction_coverage"]),
                    record["prediction_coverage"],
                ),
            }
            mismatches = [
                name
                for name, (actual, expected) in comparisons.items()
                if not _close(actual, expected)
            ]
            if mismatches:
                raise ValueError(f"metric mismatch: {', '.join(mismatches)}")
        except Exception as error:
            issues.append(
                IntegrityIssue("error", "score_content_invalid", str(error), str(path), batch_id)
            )
        return issues

    def _orphan_issues(
        self, root: Path, referenced: set[Path], *, artifact_kind: str
    ) -> list[IntegrityIssue]:
        if not root.is_dir():
            return []
        discovered = {path.resolve() for path in root.rglob("*.parquet")}
        return [
            IntegrityIssue(
                "warning",
                f"orphan_{artifact_kind}",
                f"{artifact_kind} artifact is not registered in production state",
                str(path),
            )
            for path in sorted(discovered.difference(referenced))
        ]

    def _registry_issues(self) -> list[IntegrityIssue]:
        deployment = self.state.latest_deployment(self.config.tracking.production_alias)
        if deployment is None:
            return [
                IntegrityIssue("warning", "no_deployment", "no production deployment is recorded")
            ]
        try:
            import mlflow

            client = mlflow.MlflowClient(tracking_uri=self.config.tracking.uri)
            version = client.get_model_version_by_alias(
                self.config.tracking.registered_model_name,
                self.config.tracking.production_alias,
            )
        except Exception as error:
            return [
                IntegrityIssue(
                    "error", "registry_unavailable", f"cannot resolve production alias: {error!r}"
                )
            ]
        if (
            str(version.version) != deployment.model_version
            or str(version.run_id) != deployment.model_run_id
        ):
            return [
                IntegrityIssue(
                    "error",
                    "deployment_registry_mismatch",
                    "production alias differs from the latest deployment record",
                )
            ]
        return []

    def run(self, *, verify_registry: bool = True, write_report: bool = True) -> IntegrityReport:
        predictions = self.state.prediction_history()
        scores = self.state.score_history()
        issues: list[IntegrityIssue] = []
        for record in predictions:
            issues.extend(self._prediction_issues(record))
        for record in scores:
            issues.extend(self._score_issues(record))

        prediction_paths = {Path(row["prediction_path"]).resolve() for row in predictions}
        score_paths = {Path(row["scored_path"]).resolve() for row in scores}
        issues.extend(
            self._orphan_issues(
                self.config.storage.prediction_root,
                prediction_paths,
                artifact_kind="prediction",
            )
        )
        issues.extend(
            self._orphan_issues(self.config.storage.score_root, score_paths, artifact_kind="score")
        )
        if verify_registry:
            issues.extend(self._registry_issues())

        severities = {issue.severity for issue in issues}
        status = "failed" if "error" in severities else "warning" if issues else "passed"
        report = IntegrityReport(
            generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            status=status,
            prediction_batches=len(predictions),
            score_records=len(scores),
            deployments=len(self.state.deployment_history()),
            issues=tuple(issues),
        )
        if write_report:
            _atomic_json(self.config.storage.monitoring_root / "integrity.json", report.to_dict())
        return report


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint_path: str
    manifest_path: str
    database_sha256: str
    prediction_artifacts: int
    score_artifacts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointVerification:
    status: str
    database_path: str
    database_sha256: str
    prediction_artifacts: int
    score_artifacts: int
    integrity_status: str
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        return payload


def create_control_plane_checkpoint(
    *,
    config: ProductionConfig,
    state_store: ProductionStateStore,
    destination: str | Path,
) -> CheckpointResult:
    """Create a consistent SQLite backup plus hashes of every referenced artifact."""

    audit = ProductionIntegrityAuditor(config=config, state_store=state_store).run()
    if audit.status == "failed":
        raise ValueError("cannot checkpoint production state while integrity audit fails")
    destination = Path(destination).resolve()
    checkpoint_path = state_store.backup(destination / "production.db")
    predictions = state_store.prediction_history()
    scores = state_store.score_history()
    manifest = {
        "schema_version": 1,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "database": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        },
        "predictions": [
            {
                "batch_id": row["batch_id"],
                "path": row["prediction_path"],
                "sha256": row["prediction_sha256"],
            }
            for row in predictions
        ],
        "scores": [
            {
                "batch_id": row["batch_id"],
                "path": row["scored_path"],
                "sha256": row["score_sha256"] or sha256_file(row["scored_path"]),
            }
            for row in scores
        ],
        "integrity_audit": audit.to_dict(),
    }
    manifest_path = destination / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return CheckpointResult(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(manifest_path),
        database_sha256=str(manifest["database"]["sha256"]),
        prediction_artifacts=len(predictions),
        score_artifacts=len(scores),
    )


def verify_control_plane_checkpoint(
    *,
    config: ProductionConfig,
    manifest_path: str | Path,
    verify_registry: bool = True,
) -> CheckpointVerification:
    """Open a checkpoint copy and prove it still resolves every immutable artifact."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint manifest schema")
    database_path = (manifest_path.parent / manifest["database"]["path"]).resolve()
    if not database_path.is_relative_to(manifest_path.parent):
        raise ValueError("checkpoint database path escapes the checkpoint directory")

    issues: list[str] = []
    expected_database_hash = str(manifest["database"]["sha256"])
    if not database_path.is_file():
        issues.append(f"checkpoint database is missing: {database_path}")
        actual_database_hash = "missing"
    else:
        actual_database_hash = sha256_file(database_path)
        if actual_database_hash != expected_database_hash:
            issues.append("checkpoint database checksum mismatch")

    for collection in ("predictions", "scores"):
        for artifact in manifest[collection]:
            path = Path(artifact["path"])
            if not path.is_file():
                issues.append(f"referenced artifact is missing: {path}")
            elif sha256_file(path) != artifact["sha256"]:
                issues.append(f"referenced artifact checksum mismatch: {path}")

    integrity_status = "not_run"
    if not issues:
        restored_state = ProductionStateStore(database_path)
        report = ProductionIntegrityAuditor(config=config, state_store=restored_state).run(
            verify_registry=verify_registry, write_report=False
        )
        integrity_status = report.status
        issues.extend(
            f"{issue.code}: {issue.message}" for issue in report.issues if issue.severity == "error"
        )
        if len(restored_state.prediction_history()) != len(manifest["predictions"]):
            issues.append("checkpoint prediction count differs from its manifest")
        if len(restored_state.score_history()) != len(manifest["scores"]):
            issues.append("checkpoint score count differs from its manifest")

    return CheckpointVerification(
        status="passed" if not issues else "failed",
        database_path=str(database_path),
        database_sha256=actual_database_hash,
        prediction_artifacts=len(manifest["predictions"]),
        score_artifacts=len(manifest["scores"]),
        integrity_status=integrity_status,
        issues=tuple(issues),
    )
