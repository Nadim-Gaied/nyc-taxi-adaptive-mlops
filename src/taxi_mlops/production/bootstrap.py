"""Idempotent bootstrap of a portable production registry from a native model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.state import ModelDeployment, ProductionStateStore
from taxi_mlops.tracking.fingerprints import sha256_file


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - optional installation
        raise RuntimeError(
            'MLflow is optional. Install it with: pip install -e ".[tracking]"'
        ) from error
    return mlflow


@dataclass(frozen=True)
class ProductionBootstrapResult:
    action: str
    model_name: str
    model_version: str
    model_run_id: str
    model_sha256: str
    training_start: str
    training_end: str
    feature_count: int
    reference_alias: str
    production_alias: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _bootstrap_identity(
    *,
    model_sha256: str,
    training_start: str,
    training_end: str,
    archive_ref: str,
) -> str:
    payload = {
        "archive_ref": archive_ref,
        "model_sha256": model_sha256,
        "training_end": training_end,
        "training_start": training_start,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _matching_version(client: Any, model_name: str, identity: str) -> Any | None:
    escaped_name = model_name.replace("'", "\\'")
    versions = client.search_model_versions(f"name = '{escaped_name}'")
    matches = [
        version for version in versions if version.tags.get("bootstrap_identity") == identity
    ]
    if len(matches) > 1:
        raise RuntimeError("multiple registered models have the same bootstrap identity")
    return matches[0] if matches else None


def _alias_version(mlflow: Any, client: Any, model_name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def bootstrap_production_reference(
    *,
    config: ProductionConfig,
    state_store: ProductionStateStore,
    model_path: str | Path,
    training_start: str,
    training_end: str,
    archive_ref: str = "experiment-v1.0.0",
) -> ProductionBootstrapResult:
    """Import one frozen native model into the environment-local MLflow store.

    The native model and its immutable identity are the portable seed. MLflow's
    database and artifact URIs are created inside the target environment, so a
    Windows tracking store never leaks host-only paths into a Linux container.
    """

    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    start = canonical_utc(training_start, "training_start")
    end = canonical_utc(training_end, "training_end")
    if end <= start:
        raise ValueError("training_end must be later than training_start")

    booster = lgb.Booster(model_file=str(model_path))
    feature_names = list(booster.feature_name())
    if not feature_names:
        raise ValueError("reference model does not expose any feature names")
    model_sha256 = sha256_file(model_path)
    identity = _bootstrap_identity(
        model_sha256=model_sha256,
        training_start=start.isoformat(),
        training_end=end.isoformat(),
        archive_ref=archive_ref,
    )
    current_deployment = state_store.latest_deployment(config.tracking.production_alias)
    if current_deployment is not None and (
        current_deployment.model_name != config.tracking.registered_model_name
        or current_deployment.data_fingerprint != model_sha256
        or canonical_utc(current_deployment.training_start, "deployed training_start") != start
        or canonical_utc(current_deployment.training_end, "deployed training_end") != end
    ):
        raise ValueError("production bootstrap refuses to replace an existing deployment")

    mlflow = _require_mlflow()
    mlflow.set_tracking_uri(config.tracking.uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.tracking.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            config.tracking.experiment_name,
            artifact_location=config.tracking.artifact_root.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    existing = _matching_version(client, config.tracking.registered_model_name, identity)
    protected_aliases = {
        config.tracking.reference_alias: _alias_version(
            mlflow,
            client,
            config.tracking.registered_model_name,
            config.tracking.reference_alias,
        ),
        config.tracking.production_alias: _alias_version(
            mlflow,
            client,
            config.tracking.registered_model_name,
            config.tracking.production_alias,
        ),
    }
    target_version = None if existing is None else str(existing.version)
    conflicts = {
        alias: version
        for alias, version in protected_aliases.items()
        if version is not None and version != target_version
    }
    if conflicts:
        raise ValueError(f"production bootstrap refuses to replace existing aliases: {conflicts}")
    if current_deployment is not None and (
        existing is None
        or str(existing.version) != current_deployment.model_version
        or str(existing.run_id) != current_deployment.model_run_id
    ):
        raise ValueError("production deployment and bootstrap registry lineage disagree")
    if existing is None:
        with mlflow.start_run(
            run_name=f"bootstrap-{archive_ref}",
            tags={
                "run_kind": "production_reference_bootstrap",
                "archive_ref": archive_ref,
                "bootstrap_identity": identity,
            },
        ) as run:
            mlflow.log_params(
                {
                    "training_start": start.isoformat(),
                    "training_end": end.isoformat(),
                    "feature_count": len(feature_names),
                    "model_sha256": model_sha256,
                }
            )
            mlflow.log_artifact(str(model_path), artifact_path="native")
            model_info = mlflow.lightgbm.log_model(
                booster,
                name="model",
                registered_model_name=config.tracking.registered_model_name,
                metadata={
                    "archive_ref": archive_ref,
                    "bootstrap_identity": identity,
                    "native_model_sha256": model_sha256,
                },
            )
            version = str(model_info.registered_model_version)
            run_id = run.info.run_id
        action = "registered"
    else:
        version = str(existing.version)
        run_id = str(existing.run_id)
        action = "reused"

    for key, value in {
        "archive_ref": archive_ref,
        "bootstrap_identity": identity,
        "model_sha256": model_sha256,
        "training_start": start.isoformat(),
        "training_end": end.isoformat(),
        "history_strategy": config.lifecycle.training_history,
        "feature_set": "full_v1",
        "technical_validation": "native_model_loaded",
    }.items():
        client.set_model_version_tag(
            config.tracking.registered_model_name, version, key, str(value)
        )
    client.set_registered_model_alias(
        config.tracking.registered_model_name,
        config.tracking.reference_alias,
        version,
    )
    client.set_registered_model_alias(
        config.tracking.registered_model_name,
        config.tracking.production_alias,
        version,
    )

    state_store.record_deployment(
        ModelDeployment(
            model_name=config.tracking.registered_model_name,
            model_version=version,
            model_alias=config.tracking.production_alias,
            model_run_id=run_id,
            deployed_at=end.isoformat(),
            training_start=start.isoformat(),
            training_end=end.isoformat(),
            data_fingerprint=model_sha256,
            reason="bootstrap_from_frozen_native_reference",
        )
    )
    state_store.record_event(
        "bootstrap",
        "completed",
        details={
            "action": action,
            "archive_ref": archive_ref,
            "model_version": version,
            "model_sha256": model_sha256,
        },
    )
    return ProductionBootstrapResult(
        action=action,
        model_name=config.tracking.registered_model_name,
        model_version=version,
        model_run_id=run_id,
        model_sha256=model_sha256,
        training_start=start.isoformat(),
        training_end=end.isoformat(),
        feature_count=len(feature_names),
        reference_alias=config.tracking.reference_alias,
        production_alias=config.tracking.production_alias,
    )
