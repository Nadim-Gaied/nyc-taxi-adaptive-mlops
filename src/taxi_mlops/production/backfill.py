"""Bounded, resumable production catch-up for one immutable model epoch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

import pandas as pd

from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.models import MLflowLightGBMResolver, ResolvedModel
from taxi_mlops.production.monitoring import ProductionMonitoringReporter
from taxi_mlops.production.pipeline import HourlyPredictionPipeline, HourlyScoringPipeline
from taxi_mlops.production.state import ProductionStateStore
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource


class _ModelResolver(Protocol):
    def resolve(self) -> ResolvedModel: ...


class _PinnedResolver:
    def __init__(self, model: ResolvedModel) -> None:
        self.model = model

    def resolve(self) -> ResolvedModel:
        return self.model


@dataclass(frozen=True)
class BackfillResult:
    start: str
    end: str
    hours: int
    model_name: str
    model_version: str
    predictions_created: int
    predictions_reused: int
    scores_created: int
    scores_reused: int
    elapsed_seconds: float
    monitoring: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _bounded_hours(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    maximum_hours: int,
) -> pd.DatetimeIndex:
    start_utc = canonical_utc(start, "start")
    end_utc = canonical_utc(end, "end")
    if start_utc != start_utc.floor("h") or end_utc != end_utc.floor("h"):
        raise ValueError("backfill boundaries must be exactly on UTC hours")
    if end_utc <= start_utc:
        raise ValueError("backfill end must be later than start")
    hours = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    if len(hours) > maximum_hours:
        raise ValueError(
            f"backfill requests {len(hours)} hours; configured maximum is {maximum_hours}"
        )
    return hours


class ProductionBackfillRunner:
    """Catch up a bounded interval without changing lifecycle state or model alias."""

    def __init__(
        self,
        *,
        config: ProductionConfig,
        state_store: ProductionStateStore,
        model_resolver: _ModelResolver | None = None,
    ) -> None:
        self.config = config
        self.state = state_store
        self.model_resolver = model_resolver or MLflowLightGBMResolver(
            tracking_uri=config.tracking.uri,
            model_name=config.tracking.registered_model_name,
            alias=config.tracking.production_alias,
        )

    def _validate_model_epoch(
        self, start: pd.Timestamp, end: pd.Timestamp, model: ResolvedModel
    ) -> None:
        deployments = self.state.deployment_history()
        transitions = [
            row
            for row in deployments
            if start < canonical_utc(row["deployed_at"], "deployed_at") < end
        ]
        if transitions:
            raise ValueError(
                "backfill crosses a recorded model deployment; split it into model epochs"
            )
        effective_deployments = [
            row for row in deployments if canonical_utc(row["deployed_at"], "deployed_at") <= start
        ]
        if effective_deployments:
            effective = effective_deployments[-1]
            if (
                str(effective["model_name"]) != model.name
                or str(effective["model_version"]) != model.version
            ):
                raise ValueError(
                    "resolved model does not match the model deployed at backfill start"
                )
        existing_versions = {
            str(row["model_version"])
            for row in self.state.prediction_history()
            if start <= canonical_utc(row["target_hour"], "target_hour") < end
        }
        if existing_versions.difference({model.version}):
            raise ValueError("existing predictions in the interval use another model version")

    def run(
        self,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        maximum_hours: int = 744,
    ) -> BackfillResult:
        hours = _bounded_hours(start, end, maximum_hours=maximum_hours)
        start_utc = hours[0]
        end_utc = hours[-1] + pd.Timedelta(hours=1)
        model = self.model_resolver.resolve()
        self._validate_model_epoch(start_utc, end_utc, model)
        self.state.record_event(
            "backfill",
            "started",
            details={
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
                "hours": len(hours),
                "model_version": model.version,
            },
        )
        started_at = perf_counter()
        predictions_created = 0
        predictions_reused = 0
        scores_created = 0
        scores_reused = 0
        try:
            source = MonthlyReplayDataSource(
                feature_root=self.config.storage.feature_root,
                start=start_utc,
                end=end_utc,
                timezone=self.config.timezone,
            )
            prediction_pipeline = HourlyPredictionPipeline(
                feature_root=self.config.storage.feature_root,
                output_root=self.config.storage.prediction_root,
                state_store=self.state,
                model_resolver=_PinnedResolver(model),
                timezone=self.config.timezone,
                expected_zones=self.config.expected_zones,
                data_source=source,
            )
            scoring_pipeline = HourlyScoringPipeline(
                feature_root=self.config.storage.feature_root,
                output_root=self.config.storage.score_root,
                state_store=self.state,
                timezone=self.config.timezone,
                data_source=source,
            )
            for target in hours:
                existing_prediction = self.state.prediction_for_hour(target.isoformat())
                batch = prediction_pipeline.run(target)
                if existing_prediction is None:
                    predictions_created += 1
                else:
                    predictions_reused += 1
                existing_score = self.state.score_for_batch(batch.batch_id)
                scoring_pipeline.run(target, as_of=target + pd.Timedelta(hours=1))
                if existing_score is None:
                    scores_created += 1
                else:
                    scores_reused += 1

            monitoring = ProductionMonitoringReporter(
                config=self.config, state_store=self.state
            ).build(as_of=end_utc)
            elapsed = perf_counter() - started_at
            result = BackfillResult(
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                hours=len(hours),
                model_name=model.name,
                model_version=model.version,
                predictions_created=predictions_created,
                predictions_reused=predictions_reused,
                scores_created=scores_created,
                scores_reused=scores_reused,
                elapsed_seconds=elapsed,
                monitoring=monitoring.to_dict(),
            )
            self.state.update_watermark(
                "backfill", end_utc.isoformat(), f"model-version:{model.version}"
            )
            self.state.record_event(
                "backfill",
                "completed",
                details={
                    "start": result.start,
                    "end": result.end,
                    "hours": result.hours,
                    "model_version": result.model_version,
                    "predictions_created": result.predictions_created,
                    "predictions_reused": result.predictions_reused,
                    "scores_created": result.scores_created,
                    "scores_reused": result.scores_reused,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )
            return result
        except Exception as error:
            self.state.record_event(
                "backfill",
                "failed",
                details={
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat(),
                    "error": repr(error),
                },
            )
            raise
