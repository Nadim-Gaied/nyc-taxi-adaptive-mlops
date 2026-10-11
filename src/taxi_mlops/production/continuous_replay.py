"""Advance a local, time-faithful replay one observed hour at a time."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Protocol

import pandas as pd

from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.models import MLflowLightGBMResolver, ResolvedModel
from taxi_mlops.production.pipeline import HourlyPredictionPipeline, HourlyScoringPipeline
from taxi_mlops.production.state import ProductionStateStore
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource


@dataclass(frozen=True)
class ReplayStep:
    virtual_clock: str
    scored_hour: str | None
    forecast_hour: str


class _ModelResolver(Protocol):
    def resolve(self) -> ResolvedModel: ...


class ContinuousReplay:
    """Release labels after their hour ends, then forecast the next hour."""

    def __init__(
        self,
        config: ProductionConfig,
        *,
        start: str,
        end: str,
        model_resolver: _ModelResolver | None = None,
    ) -> None:
        self.config = config
        self.start = canonical_utc(start, "start")
        self.end = canonical_utc(end, "end")
        if self.start != self.start.floor("h") or self.end != self.end.floor("h"):
            raise ValueError("replay boundaries must be exact UTC hours")
        if self.end <= self.start:
            raise ValueError("replay end must follow start")
        self.state = ProductionStateStore(config.storage.state_db)
        self.source = MonthlyReplayDataSource(
            feature_root=config.storage.feature_root,
            start=self.start,
            end=self.end,
            timezone=config.timezone,
        )
        self.prediction = HourlyPredictionPipeline(
            feature_root=config.storage.feature_root,
            output_root=config.storage.prediction_root,
            state_store=self.state,
            model_resolver=model_resolver
            or MLflowLightGBMResolver(
                tracking_uri=config.tracking.uri,
                model_name=config.tracking.registered_model_name,
                alias=config.tracking.production_alias,
            ),
            timezone=config.timezone,
            expected_zones=config.expected_zones,
            data_source=self.source,
        )
        self.scoring = HourlyScoringPipeline(
            feature_root=config.storage.feature_root,
            output_root=config.storage.score_root,
            state_store=self.state,
            timezone=config.timezone,
            data_source=self.source,
        )

    def _cursor(self) -> pd.Timestamp:
        history = self.state.prediction_history()
        if not history:
            return self.start
        latest = max(canonical_utc(row["target_hour"], "target_hour") for row in history)
        if latest < self.start or latest >= self.end:
            raise ValueError("existing replay state lies outside the requested interval")
        return latest

    def run(self, *, interval_seconds: float = 5, max_ticks: int | None = None):
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be nonnegative")
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max_ticks must be positive")
        cursor = self._cursor()
        self.prediction.run(cursor)
        yield ReplayStep(cursor.isoformat(), None, cursor.isoformat())
        ticks = 0
        while cursor + pd.Timedelta(hours=1) < self.end:
            if max_ticks is not None and ticks >= max_ticks:
                break
            if interval_seconds:
                sleep(interval_seconds)
            next_hour = cursor + pd.Timedelta(hours=1)
            self.scoring.run(cursor, as_of=next_hour)
            self.state.record_event(
                "replay_source",
                "labels_released",
                details={"target_hour": cursor.isoformat(), "virtual_clock": next_hour.isoformat()},
            )
            self.prediction.run(next_hour)
            yield ReplayStep(next_hour.isoformat(), cursor.isoformat(), next_hour.isoformat())
            cursor = next_hour
            ticks += 1
