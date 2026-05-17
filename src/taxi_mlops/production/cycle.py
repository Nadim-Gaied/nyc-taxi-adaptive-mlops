"""One deterministic production control cycle for an hourly scheduler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import pandas as pd

from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.lifecycle import ScheduledTrainingPipeline
from taxi_mlops.production.models import MLflowLightGBMResolver
from taxi_mlops.production.monitoring import ProductionMonitoringReporter
from taxi_mlops.production.pipeline import HourlyPredictionPipeline, HourlyScoringPipeline
from taxi_mlops.production.state import ProductionStateStore


class _Runnable(Protocol):
    def run(self, *args: Any, **kwargs: Any) -> Any: ...


class _MonitoringReporter(Protocol):
    def build(self, *, as_of: str | pd.Timestamp) -> Any: ...


@dataclass(frozen=True)
class ProductionCycleResult:
    as_of: str
    prediction_target: str
    scoring_target: str
    training: dict[str, object]
    scoring: dict[str, object]
    prediction: dict[str, object]
    monitoring: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cycle_hours(as_of: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return current prediction hour and completed prior scoring hour."""

    cutoff = canonical_utc(as_of, "as_of")
    if cutoff != cutoff.floor("h"):
        raise ValueError("production cycle as_of must be exactly on an hour boundary")
    return cutoff, cutoff - pd.Timedelta(hours=1)


class ProductionCycleRunner:
    """Coordinate lifecycle steps without coupling them to a scheduler vendor."""

    def __init__(
        self,
        *,
        config: ProductionConfig,
        state_store: ProductionStateStore,
        training_pipeline: _Runnable | None = None,
        prediction_pipeline: _Runnable | None = None,
        scoring_pipeline: _Runnable | None = None,
        monitoring_reporter: _MonitoringReporter | None = None,
    ) -> None:
        self.config = config
        self.state = state_store
        self.training = training_pipeline or ScheduledTrainingPipeline(
            config=config, state_store=state_store
        )
        self.prediction = prediction_pipeline or HourlyPredictionPipeline(
            feature_root=config.storage.feature_root,
            output_root=config.storage.prediction_root,
            state_store=state_store,
            model_resolver=MLflowLightGBMResolver(
                tracking_uri=config.tracking.uri,
                model_name=config.tracking.registered_model_name,
                alias=config.tracking.production_alias,
            ),
            timezone=config.timezone,
            expected_zones=config.expected_zones,
        )
        self.scoring = scoring_pipeline or HourlyScoringPipeline(
            feature_root=config.storage.feature_root,
            output_root=config.storage.score_root,
            state_store=state_store,
            timezone=config.timezone,
        )
        self.monitoring = monitoring_reporter or ProductionMonitoringReporter(
            config=config, state_store=state_store
        )

    def run(
        self,
        as_of: str | pd.Timestamp,
        *,
        allow_missing_previous_prediction: bool = False,
    ) -> ProductionCycleResult:
        prediction_target, scoring_target = cycle_hours(as_of)
        cutoff = prediction_target
        self.state.record_event(
            "cycle",
            "started",
            details={
                "as_of": cutoff.isoformat(),
                "prediction_target": prediction_target.isoformat(),
                "scoring_target": scoring_target.isoformat(),
            },
        )
        try:
            training = self.training.run(cutoff)
            previous = self.state.prediction_for_hour(scoring_target.isoformat())
            if previous is None:
                if not allow_missing_previous_prediction:
                    raise ValueError(
                        "previous-hour prediction is missing; backfill it or explicitly "
                        "allow bootstrap startup"
                    )
                scoring_payload: dict[str, object] = {
                    "action": "skipped_bootstrap",
                    "target_hour": scoring_target.isoformat(),
                    "reason": "previous_hour_prediction_missing",
                }
                self.state.record_event(
                    "score",
                    "skipped",
                    details=scoring_payload,
                )
            else:
                scoring_payload = asdict(self.scoring.run(scoring_target, as_of=cutoff))

            prediction = self.prediction.run(prediction_target)
            monitoring = self.monitoring.build(as_of=cutoff)
            result = ProductionCycleResult(
                as_of=cutoff.isoformat(),
                prediction_target=prediction_target.isoformat(),
                scoring_target=scoring_target.isoformat(),
                training=training.to_dict(),
                scoring=scoring_payload,
                prediction=asdict(prediction),
                monitoring=monitoring.to_dict(),
            )
            self.state.update_watermark("cycle", cutoff.isoformat(), prediction.feature_fingerprint)
            self.state.record_event(
                "cycle",
                "completed",
                batch_id=prediction.batch_id,
                details={
                    "as_of": cutoff.isoformat(),
                    "training_action": result.training["action"],
                    "forecast_status": result.monitoring["forecast_status"],
                },
            )
            return result
        except Exception as error:
            self.state.record_event(
                "cycle",
                "failed",
                details={"as_of": cutoff.isoformat(), "error": repr(error)},
            )
            raise
