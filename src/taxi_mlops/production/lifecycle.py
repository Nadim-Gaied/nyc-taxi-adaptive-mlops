"""Scheduled production training with technical validation and MLflow registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi_mlops.ingestion.inventory import month_sequence
from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.models import MLflowLightGBMResolver
from taxi_mlops.production.state import ModelDeployment, ProductionStateStore
from taxi_mlops.replay import ModelRegistry, ReplayModelStore, shift_local_calendar_days
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource
from taxi_mlops.tracking.fingerprints import fingerprint_files, sha256_file


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - optional installation
        raise RuntimeError(
            'MLflow is optional. Install it with: pip install -e ".[tracking]"'
        ) from error
    return mlflow


@dataclass(frozen=True)
class TrainingResult:
    action: str
    as_of: str
    due_at: str
    model_name: str
    model_version: str
    model_run_id: str
    training_rows: int
    training_seconds: float
    data_fingerprint: str
    technical_validation_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ScheduledTrainingPipeline:
    """Implement the research-backed fixed-28-day production lifecycle."""

    def __init__(
        self,
        *,
        config: ProductionConfig,
        state_store: ProductionStateStore,
        model_parameters: dict[str, object] | None = None,
    ) -> None:
        self.config = config
        self.state = state_store
        self.model_parameters = model_parameters

    def _client(self) -> tuple[Any, Any]:
        mlflow = _require_mlflow()
        mlflow.set_tracking_uri(self.config.tracking.uri)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.config.tracking.experiment_name)
        if experiment is None:
            artifact_location = self.config.tracking.artifact_root.resolve().as_uri()
            experiment_id = client.create_experiment(
                self.config.tracking.experiment_name,
                artifact_location=artifact_location,
            )
            experiment = client.get_experiment(experiment_id)
        mlflow.set_experiment(self.config.tracking.experiment_name)
        return mlflow, client

    def _bootstrap_deployment(self) -> ModelDeployment:
        alias = self.config.tracking.production_alias
        existing = self.state.latest_deployment(alias)
        if existing is not None:
            return existing
        mlflow, client = self._client()
        model_name = self.config.tracking.registered_model_name
        try:
            version = client.get_model_version_by_alias(model_name, alias)
        except mlflow.exceptions.MlflowException:
            reference = client.get_model_version_by_alias(
                model_name, self.config.tracking.reference_alias
            )
            client.set_registered_model_alias(model_name, alias, reference.version)
            version = client.get_model_version_by_alias(model_name, alias)
        tags = dict(version.tags)
        training_start = tags.get("training_start")
        training_end = tags.get("training_end")
        if not training_start or not training_end:
            raise ValueError("production bootstrap model lacks training interval tags")
        deployment = ModelDeployment(
            model_name=model_name,
            model_version=str(version.version),
            model_alias=alias,
            model_run_id=str(version.run_id),
            deployed_at=canonical_utc(training_end, "training_end").isoformat(),
            training_start=canonical_utc(training_start, "training_start").isoformat(),
            training_end=canonical_utc(training_end, "training_end").isoformat(),
            data_fingerprint=tags.get("model_sha256", "historical-import"),
            reason="bootstrap_from_frozen_accuracy_reference",
        )
        return self.state.record_deployment(deployment)

    def _feature_files(self, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
        local_start = start.tz_convert(self.config.timezone)
        local_last = (end - pd.Timedelta(nanoseconds=1)).tz_convert(self.config.timezone)
        months = month_sequence(local_start.strftime("%Y-%m"), local_last.strftime("%Y-%m"))
        paths = [
            self.config.storage.feature_root / month[:4] / f"features_{month}.parquet"
            for month in months
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing production feature partitions: {missing}")
        return paths

    def run(self, as_of: str | pd.Timestamp, *, force: bool = False) -> TrainingResult:
        cutoff = canonical_utc(as_of, "as_of")
        current = self._bootstrap_deployment()
        due = shift_local_calendar_days(
            current.deployed_at,
            days=self.config.lifecycle.retrain_every_calendar_days,
            timezone=self.config.timezone,
        )
        if cutoff < due and not force:
            return TrainingResult(
                action="not_due",
                as_of=cutoff.isoformat(),
                due_at=due.isoformat(),
                model_name=current.model_name,
                model_version=current.model_version,
                model_run_id=current.model_run_id,
                training_rows=0,
                training_seconds=0.0,
                data_fingerprint=current.data_fingerprint,
                technical_validation_passed=True,
            )

        self.state.record_event("train", "started", details={"as_of": cutoff.isoformat()})
        try:
            history_start = shift_local_calendar_days(
                cutoff, days=-365, timezone=self.config.timezone
            )
            feature_files = self._feature_files(history_start, cutoff)
            data_fingerprint = fingerprint_files(
                feature_files, root=self.config.storage.feature_root
            )
            resolver = MLflowLightGBMResolver(
                tracking_uri=self.config.tracking.uri,
                model_name=self.config.tracking.registered_model_name,
                alias=self.config.tracking.production_alias,
            )
            current_model = resolver.resolve()
            source = MonthlyReplayDataSource(
                feature_root=self.config.storage.feature_root,
                start=history_start,
                end=cutoff,
                timezone=self.config.timezone,
            )
            training_data = source.observations_as_of(cutoff, start=history_start)
            if training_data.empty or training_data["pickup_hour"].min() > history_start:
                raise ValueError("complete recent-365-day training history is unavailable")

            identity_payload = {
                "cutoff": cutoff.isoformat(),
                "data_fingerprint": data_fingerprint,
                "feature_columns": current_model.feature_columns,
                "history": self.config.lifecycle.training_history,
                "model_parameters": self.model_parameters or {},
            }
            training_identity = hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            model_id = f"production__recent_365d__{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
            registry = ModelRegistry()
            model_store = ReplayModelStore(self.config.storage.model_root, registry)
            outcome = model_store.train(
                training_data,
                model_id=model_id,
                as_of=cutoff,
                history_strategy=self.config.lifecycle.training_history,
                feature_set="full_v1",
                feature_columns=current_model.feature_columns,
                timezone=self.config.timezone,
                model_parameters=self.model_parameters,
            )

            validation_hour = cutoff - pd.Timedelta(hours=1)
            validation_features = source.prediction_features(validation_hour, as_of=validation_hour)
            prediction, _ = model_store.predict(model_id, validation_features)
            reloaded = ReplayModelStore(self.config.storage.model_root, registry)
            reloaded_prediction, _ = reloaded.predict(model_id, validation_features)
            maximum_reload_difference = float(
                np.max(np.abs(prediction.to_numpy() - reloaded_prediction.to_numpy()))
            )
            technical_pass = bool(
                len(prediction) == self.config.expected_zones
                and np.isfinite(prediction.to_numpy()).all()
                and prediction.ge(0).all()
                and maximum_reload_difference <= 1e-12
            )
            if not technical_pass:
                raise ValueError("trained model failed technical validation")

            mlflow, client = self._client()
            with mlflow.start_run(
                run_name=model_id,
                tags={
                    "run_kind": "production_training",
                    "training_identity": training_identity,
                    "lifecycle_strategy": self.config.lifecycle.strategy,
                },
            ) as run:
                mlflow.log_params(
                    {
                        "history_strategy": self.config.lifecycle.training_history,
                        "feature_set": "full_v1",
                        "training_start": outcome.record.training_start,
                        "training_end": outcome.record.training_end,
                        "data_fingerprint": data_fingerprint,
                        **(self.model_parameters or {}),
                    }
                )
                mlflow.log_metrics(
                    {
                        "training_rows": outcome.training_rows,
                        "training_seconds": outcome.training_seconds,
                        "technical_validation_rows": len(prediction),
                        "technical_reload_max_abs_difference": maximum_reload_difference,
                    }
                )
                mlflow.log_artifact(outcome.record.artifact_path, artifact_path="native")
                metadata_path = Path(outcome.record.artifact_path).with_suffix(".json")
                mlflow.log_artifact(str(metadata_path), artifact_path="native")
                booster = lgb.Booster(model_file=outcome.record.artifact_path)
                model_info = mlflow.lightgbm.log_model(
                    booster,
                    name="model",
                    registered_model_name=self.config.tracking.registered_model_name,
                    metadata={
                        "training_identity": training_identity,
                        "data_fingerprint": data_fingerprint,
                        "native_model_sha256": sha256_file(outcome.record.artifact_path),
                    },
                )
                version = str(model_info.registered_model_version)
                for key, value in {
                    "training_start": outcome.record.training_start,
                    "training_end": outcome.record.training_end,
                    "history_strategy": outcome.record.history_strategy,
                    "feature_set": outcome.record.feature_set,
                    "data_fingerprint": data_fingerprint,
                    "technical_validation": "passed",
                }.items():
                    client.set_model_version_tag(
                        self.config.tracking.registered_model_name, version, key, str(value)
                    )
                client.set_registered_model_alias(
                    self.config.tracking.registered_model_name,
                    self.config.tracking.production_alias,
                    version,
                )
                deployment = ModelDeployment(
                    model_name=self.config.tracking.registered_model_name,
                    model_version=version,
                    model_alias=self.config.tracking.production_alias,
                    model_run_id=run.info.run_id,
                    deployed_at=cutoff.isoformat(),
                    training_start=canonical_utc(
                        outcome.record.training_start, "training_start"
                    ).isoformat(),
                    training_end=canonical_utc(
                        outcome.record.training_end, "training_end"
                    ).isoformat(),
                    data_fingerprint=data_fingerprint,
                    reason="scheduled_28_day_retraining_technical_validation_passed",
                )
                self.state.record_deployment(deployment)
                mlflow.set_tag("registered_model_version", version)

            self.state.update_watermark("train", cutoff.isoformat(), data_fingerprint)
            self.state.record_event(
                "train",
                "completed",
                details={
                    "model_version": version,
                    "training_rows": outcome.training_rows,
                    "training_seconds": outcome.training_seconds,
                    "technical_validation_passed": True,
                },
            )
            return TrainingResult(
                action="trained_and_deployed",
                as_of=cutoff.isoformat(),
                due_at=due.isoformat(),
                model_name=self.config.tracking.registered_model_name,
                model_version=version,
                model_run_id=run.info.run_id,
                training_rows=outcome.training_rows,
                training_seconds=outcome.training_seconds,
                data_fingerprint=data_fingerprint,
                technical_validation_passed=True,
            )
        except Exception as error:
            self.state.record_event(
                "train", "failed", details={"as_of": cutoff.isoformat(), "error": repr(error)}
            )
            raise
