"""Idempotent hourly prediction and delayed-label scoring pipelines."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

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
    ) -> None:
        self.feature_root = Path(feature_root)
        self.output_root = Path(output_root)
        self.state = state_store
        self.model_resolver = model_resolver
        self.timezone = timezone
        self.expected_zones = expected_zones

    def run(self, target_hour: str | pd.Timestamp) -> PredictionBatch:
        target = canonical_utc(target_hour, "target_hour")
        target_text = target.isoformat()
        existing = self.state.prediction_for_hour(target_text)
        if existing is not None:
            if not Path(existing.prediction_path).is_file():
                raise FileNotFoundError(existing.prediction_path)
            if sha256_file(existing.prediction_path) != existing.prediction_sha256:
                raise ValueError("persisted prediction artifact checksum has changed")
            return existing

        self.state.record_event("predict", "started", details={"target_hour": target_text})
        try:
            model = self.model_resolver.resolve()
            source = MonthlyReplayDataSource(
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
            output["prediction"] = model.predict(features)
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
            output["feature_fingerprint"] = feature_fingerprint
            path = _prediction_path(self.output_root, target).resolve()
            if path.exists():
                raise FileExistsError(
                    f"untracked prediction artifact already exists for target: {path}"
                )
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
            self.state.record_event("predict", "completed", batch_id=batch_id)
            return batch
        except Exception as error:
            self.state.record_event(
                "predict",
                "failed",
                details={"target_hour": target_text, "error": repr(error)},
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
    ) -> None:
        self.feature_root = Path(feature_root)
        self.output_root = Path(output_root)
        self.state = state_store
        self.timezone = timezone

    def run(
        self,
        target_hour: str | pd.Timestamp,
        *,
        as_of: str | pd.Timestamp,
    ) -> ScoreRecord:
        target = canonical_utc(target_hour, "target_hour")
        cutoff = canonical_utc(as_of, "as_of")
        target_text = target.isoformat()
        batch = self.state.prediction_for_hour(target_text)
        if batch is None:
            raise ValueError(f"no prediction batch exists for {target_text}")
        existing = self.state.score_for_batch(batch.batch_id)
        if existing is not None:
            if not Path(existing.scored_path).is_file():
                raise FileNotFoundError(existing.scored_path)
            return existing
        if cutoff < target + pd.Timedelta(hours=1):
            raise ValueError("target label is not complete at the scoring cutoff")

        self.state.record_event("score", "started", batch_id=batch.batch_id)
        try:
            predictions = pd.read_parquet(batch.prediction_path)
            source = MonthlyReplayDataSource(
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
            scored_path = (
                self.output_root
                / target.strftime("%Y")
                / target.strftime("%m")
                / target.strftime("%d")
                / f"scored_{target.strftime('%Y%m%dT%H%M%SZ')}.parquet"
            ).resolve()
            if scored_path.exists():
                raise FileExistsError(f"untracked score artifact already exists: {scored_path}")
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
            )
            self.state.record_score(score)
            self.state.update_watermark("score", target_text, batch.feature_fingerprint)
            self.state.record_event("score", "completed", batch_id=batch.batch_id)
            return score
        except Exception as error:
            self.state.record_event(
                "score", "failed", batch_id=batch.batch_id, details={"error": repr(error)}
            )
            raise
