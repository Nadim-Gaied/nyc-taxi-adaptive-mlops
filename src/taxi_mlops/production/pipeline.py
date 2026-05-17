"""Idempotent hourly prediction and delayed-label scoring pipelines."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from taxi_mlops.evaluation import regression_metrics
from taxi_mlops.production.contracts import (
    canonical_utc,
    dataframe_fingerprint,
    validate_prediction_features,
    validate_predictions,
)
from taxi_mlops.production.models import MLflowLightGBMResolver
from taxi_mlops.production.state import PredictionBatch, ProductionStateStore, ScoreRecord
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource
from taxi_mlops.tracking.fingerprints import sha256_file


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prediction_path(root: Path, target: pd.Timestamp) -> Path:
    return (
        root
        / target.strftime("%Y")
        / target.strftime("%m")
        / target.strftime("%d")
        / f"predictions_{target.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    )


def _score_path(root: Path, target: pd.Timestamp) -> Path:
    return (
        root
        / target.strftime("%Y")
        / target.strftime("%m")
        / target.strftime("%d")
        / f"scored_{target.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    )


def _single_value(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame:
        raise ValueError(f"recovery artifact is missing lineage column: {column}")
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"recovery artifact has ambiguous {column} lineage")
    return values.iloc[0]


def _recover_prediction(
    path: Path,
    *,
    target: pd.Timestamp,
    state: ProductionStateStore,
    expected_zones: int | None,
) -> PredictionBatch:
    output = pd.read_parquet(path)
    required = {"pickup_hour", "pickup_zone_id", "prediction"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"orphan prediction artifact is missing: {sorted(missing)}")
    output["pickup_hour"] = pd.to_datetime(output["pickup_hour"], utc=True, errors="raise")
    if not output["pickup_hour"].eq(target).all():
        raise ValueError("orphan prediction artifact contains the wrong target hour")
    if output.duplicated(["pickup_hour", "pickup_zone_id"]).any():
        raise ValueError("orphan prediction artifact contains duplicate keys")
    if expected_zones is not None and len(output) != expected_zones:
        raise ValueError("orphan prediction artifact has incomplete zone coverage")
    values = output["prediction"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("orphan prediction artifact contains invalid predictions")

    model_name = str(_single_value(output, "model_name"))
    model_version = str(_single_value(output, "model_version"))
    fingerprint = str(_single_value(output, "feature_fingerprint"))
    identity = "|".join([target.isoformat(), model_name, model_version, fingerprint])
    expected_batch_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    batch_id = str(_single_value(output, "batch_id"))
    if batch_id != expected_batch_id:
        raise ValueError("orphan prediction artifact has an invalid batch identity")
    model_uri = (
        str(_single_value(output, "model_uri"))
        if "model_uri" in output
        else f"models:/{model_name}/{model_version}"
    )
    batch = PredictionBatch(
        batch_id=batch_id,
        target_hour=target.isoformat(),
        generated_at=str(_single_value(output, "generated_at")),
        model_name=model_name,
        model_version=model_version,
        model_alias=str(_single_value(output, "model_alias")),
        model_run_id=str(_single_value(output, "model_run_id")),
        model_uri=model_uri,
        feature_fingerprint=fingerprint,
        prediction_path=str(path.resolve()),
        prediction_sha256=sha256_file(path),
        row_count=len(output),
    )
    state.record_prediction(batch)
    state.update_watermark("predict", target.isoformat(), fingerprint)
    state.record_event(
        "predict",
        "recovered",
        batch_id=batch_id,
        details={"target_hour": target.isoformat(), "artifact": str(path.resolve())},
    )
    return batch


def _recover_score(
    path: Path,
    *,
    target: pd.Timestamp,
    cutoff: pd.Timestamp,
    batch: PredictionBatch,
    state: ProductionStateStore,
) -> ScoreRecord:
    scored = pd.read_parquet(path)
    if "batch_id" not in scored or set(scored["batch_id"].astype(str)) != {batch.batch_id}:
        raise ValueError("orphan score artifact has conflicting batch lineage")
    scored["pickup_hour"] = pd.to_datetime(scored["pickup_hour"], utc=True, errors="raise")
    if not scored["pickup_hour"].eq(target).all():
        raise ValueError("orphan score artifact contains the wrong target hour")
    if len(scored) != batch.row_count:
        raise ValueError("orphan score artifact has incomplete observation coverage")
    metrics = regression_metrics(scored)
    scored_at = (
        canonical_utc(_single_value(scored, "scored_at"), "scored_at")
        if "scored_at" in scored
        else cutoff
    )
    score = ScoreRecord(
        batch_id=batch.batch_id,
        scored_at=scored_at.isoformat(),
        scored_path=str(path.resolve()),
        observations=int(metrics["observations"]),
        actual_pickups=float(scored["pickup_count"].sum()),
        predicted_pickups=float(scored["prediction"].sum()),
        mae=float(metrics["mae"]),
        wape=None if metrics["wape"] is None else float(metrics["wape"]),
        mean_bias=float(metrics["mean_bias"]),
        prediction_coverage=float(metrics["prediction_coverage"]),
        score_sha256=sha256_file(path),
    )
    state.record_score(score)
    state.update_watermark("score", target.isoformat(), batch.feature_fingerprint)
    state.record_event(
        "score",
        "recovered",
        batch_id=batch.batch_id,
        details={"target_hour": target.isoformat(), "artifact": str(path.resolve())},
    )
    return score


class HourlyPredictionPipeline:
    def __init__(
        self,
        *,
        feature_root: str | Path,
        output_root: str | Path,
        state_store: ProductionStateStore,
        model_resolver: MLflowLightGBMResolver,
        timezone: str = "America/New_York",
        expected_zones: int | None = 263,
        data_source: MonthlyReplayDataSource | None = None,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.output_root = Path(output_root)
        self.state = state_store
        self.model_resolver = model_resolver
        self.timezone = timezone
        self.expected_zones = expected_zones
        self.data_source = data_source

    def run(self, target_hour: str | pd.Timestamp) -> PredictionBatch:
        started_at = perf_counter()
        target = canonical_utc(target_hour, "target_hour")
        target_text = target.isoformat()
        existing = self.state.prediction_for_hour(target_text)
        if existing is not None:
            if not Path(existing.prediction_path).is_file():
                raise FileNotFoundError(existing.prediction_path)
            if sha256_file(existing.prediction_path) != existing.prediction_sha256:
                raise ValueError("persisted prediction artifact checksum has changed")
            return existing

        path = _prediction_path(self.output_root, target).resolve()
        if path.is_file():
            return _recover_prediction(
                path,
                target=target,
                state=self.state,
                expected_zones=self.expected_zones,
            )

        self.state.record_event("predict", "started", details={"target_hour": target_text})
        try:
            model = self.model_resolver.resolve()
            source = self.data_source or MonthlyReplayDataSource(
                feature_root=self.feature_root,
                start=target,
                end=target + pd.Timedelta(hours=1),
                timezone=self.timezone,
            )
            raw_features = source.prediction_features(target, as_of=target)
            features = validate_prediction_features(
                raw_features,
                target_hour=target,
                feature_columns=model.feature_columns,
                expected_zones=self.expected_zones,
            )
            fingerprint_columns = list(
                dict.fromkeys(["pickup_hour", "pickup_zone_id", *model.feature_columns])
            )
            feature_fingerprint = dataframe_fingerprint(features, fingerprint_columns)
            output = features[["pickup_hour", "pickup_zone_id"]].copy()
            inference_started_at = perf_counter()
            output["prediction"] = model.predict(features)
            inference_seconds = perf_counter() - inference_started_at
            output = validate_predictions(output, expected_keys=features)
            identity = "|".join([target_text, model.name, model.version, feature_fingerprint])
            batch_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            generated_at = pd.Timestamp.now(tz="UTC").isoformat()
            output["batch_id"] = batch_id
            output["generated_at"] = generated_at
            output["model_name"] = model.name
            output["model_version"] = model.version
            output["model_alias"] = model.alias
            output["model_run_id"] = model.run_id
            output["model_uri"] = model.source_uri
            output["feature_fingerprint"] = feature_fingerprint
            _atomic_parquet(output, path)
            batch = PredictionBatch(
                batch_id=batch_id,
                target_hour=target_text,
                generated_at=generated_at,
                model_name=model.name,
                model_version=model.version,
                model_alias=model.alias,
                model_run_id=model.run_id,
                model_uri=model.source_uri,
                feature_fingerprint=feature_fingerprint,
                prediction_path=str(path),
                prediction_sha256=sha256_file(path),
                row_count=len(output),
            )
            self.state.record_prediction(batch)
            self.state.update_watermark("predict", target_text, feature_fingerprint)
            self.state.record_event(
                "predict",
                "completed",
                batch_id=batch_id,
                details={
                    "target_hour": target_text,
                    "row_count": len(output),
                    "model_version": model.version,
                    "inference_seconds": inference_seconds,
                    "pipeline_seconds": perf_counter() - started_at,
                },
            )
            return batch
        except Exception as error:
            self.state.record_event(
                "predict",
                "failed",
                details={
                    "target_hour": target_text,
                    "error": repr(error),
                    "pipeline_seconds": perf_counter() - started_at,
                },
            )
            raise


class HourlyScoringPipeline:
    def __init__(
        self,
        *,
        feature_root: str | Path,
        output_root: str | Path,
        state_store: ProductionStateStore,
        timezone: str = "America/New_York",
        data_source: MonthlyReplayDataSource | None = None,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.output_root = Path(output_root)
        self.state = state_store
        self.timezone = timezone
        self.data_source = data_source

    def run(
        self,
        target_hour: str | pd.Timestamp,
        *,
        as_of: str | pd.Timestamp,
    ) -> ScoreRecord:
        started_at = perf_counter()
        target = canonical_utc(target_hour, "target_hour")
        cutoff = canonical_utc(as_of, "as_of")
        target_text = target.isoformat()
        batch = self.state.prediction_for_hour(target_text)
        if batch is None:
            raise ValueError(f"no prediction batch exists for {target_text}")
        existing = self.state.score_for_batch(batch.batch_id)
        if existing is not None:
            existing_path = Path(existing.scored_path)
            if not existing_path.is_file():
                raise FileNotFoundError(existing.scored_path)
            checksum = sha256_file(existing_path)
            if existing.score_sha256 is None:
                return self.state.set_score_checksum(batch.batch_id, checksum)
            if existing.score_sha256 != checksum:
                raise ValueError("persisted score artifact checksum has changed")
            return existing
        if cutoff < target + pd.Timedelta(hours=1):
            raise ValueError("target label is not complete at the scoring cutoff")

        scored_path = _score_path(self.output_root, target).resolve()
        if scored_path.is_file():
            return _recover_score(
                scored_path,
                target=target,
                cutoff=cutoff,
                batch=batch,
                state=self.state,
            )

        self.state.record_event("score", "started", batch_id=batch.batch_id)
        try:
            predictions = pd.read_parquet(batch.prediction_path)
            source = self.data_source or MonthlyReplayDataSource(
                feature_root=self.feature_root,
                start=target,
                end=target + pd.Timedelta(hours=1),
                timezone=self.timezone,
            )
            labels = source.labels_between(target, target + pd.Timedelta(hours=1), as_of=cutoff)
            scored = predictions.merge(
                labels,
                on=["pickup_hour", "pickup_zone_id"],
                how="left",
                validate="1:1",
            )
            if scored["pickup_count"].isna().any():
                raise ValueError("scoring labels do not cover the prediction grid")
            metrics = regression_metrics(scored)
            scored["scored_at"] = cutoff.isoformat()
            _atomic_parquet(scored, scored_path)
            score = ScoreRecord(
                batch_id=batch.batch_id,
                scored_at=cutoff.isoformat(),
                scored_path=str(scored_path),
                observations=int(metrics["observations"]),
                actual_pickups=float(scored["pickup_count"].sum()),
                predicted_pickups=float(scored["prediction"].sum()),
                mae=float(metrics["mae"]),
                wape=None if metrics["wape"] is None else float(metrics["wape"]),
                mean_bias=float(metrics["mean_bias"]),
                prediction_coverage=float(metrics["prediction_coverage"]),
                score_sha256=sha256_file(scored_path),
            )
            self.state.record_score(score)
            self.state.update_watermark("score", target_text, batch.feature_fingerprint)
            self.state.record_event(
                "score",
                "completed",
                batch_id=batch.batch_id,
                details={
                    "target_hour": target_text,
                    "observations": score.observations,
                    "pipeline_seconds": perf_counter() - started_at,
                },
            )
            return score
        except Exception as error:
            self.state.record_event(
                "score",
                "failed",
                batch_id=batch.batch_id,
                details={
                    "target_hour": target_text,
                    "error": repr(error),
                    "pipeline_seconds": perf_counter() - started_at,
                },
            )
            raise
