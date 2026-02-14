"""Event ordering for historical production replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from taxi_mlops.replay.clock import ReplayClock
from taxi_mlops.replay.registry import ModelRegistry
from taxi_mlops.replay.source import PredictionLedger, ReplayDataSource


@dataclass(frozen=True)
class ReplayContext:
    now: pd.Timestamp
    source: ReplayDataSource
    ledger: PredictionLedger
    registry: ModelRegistry

    def history(self, *, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
        return self.source.observations_as_of(self.now, start=start)


class ReplayStrategy(Protocol):
    """Interface shared by never, periodic, triggered, and adaptive policies."""

    name: str

    def initialize(self, context: ReplayContext) -> None: ...

    def observe(self, context: ReplayContext, scored: pd.DataFrame) -> None: ...

    def decide(self, context: ReplayContext) -> None: ...

    def predict(
        self, context: ReplayContext, features: pd.DataFrame
    ) -> pd.DataFrame: ...


class ReplayEngine:
    """Release labels, decide, and predict in production-safe chronological order."""

    def __init__(
        self,
        *,
        clock: ReplayClock,
        source: ReplayDataSource,
        strategy: ReplayStrategy,
        registry: ModelRegistry | None = None,
        ledger: PredictionLedger | None = None,
    ) -> None:
        self.clock = clock
        self.source = source
        self.strategy = strategy
        self.registry = registry or ModelRegistry()
        self.ledger = ledger or PredictionLedger()
        self._events: list[dict[str, object]] = []

    def _context(self) -> ReplayContext:
        return ReplayContext(
            now=self.clock.now,
            source=self.source,
            ledger=self.ledger,
            registry=self.registry,
        )

    def run(self) -> pd.DataFrame:
        self.strategy.initialize(self._context())
        while not self.clock.finished:
            context = self._context()
            scored = self.ledger.score_available(self.source, as_of=context.now)
            if not scored.empty:
                self.strategy.observe(context, scored)
                self._events.append(
                    {"timestamp": context.now, "event": "labels_scored", "rows": len(scored)}
                )
            if self.clock.decision_due():
                self.strategy.decide(context)
                self._events.append(
                    {"timestamp": context.now, "event": "lifecycle_decision", "rows": 0}
                )
                self.clock.mark_decision()
            features = self.source.prediction_features(
                context.now, as_of=context.now
            )
            if not features.empty:
                predictions = self.strategy.predict(context, features)
                required = {"pickup_hour", "pickup_zone_id", "prediction", "model_id"}
                missing = required.difference(predictions.columns)
                if missing:
                    raise ValueError(
                        "strategy predictions are missing columns: "
                        f"{', '.join(sorted(missing))}"
                    )
                expected_keys = features[["pickup_hour", "pickup_zone_id"]].sort_values(
                    ["pickup_hour", "pickup_zone_id"]
                ).reset_index(drop=True)
                prediction_keys = predictions[
                    ["pickup_hour", "pickup_zone_id"]
                ].sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(drop=True)
                if not prediction_keys.equals(expected_keys):
                    raise ValueError("strategy did not predict the complete target grid")
                if predictions["model_id"].nunique() != 1:
                    raise ValueError("one prediction batch cannot mix model IDs")
                self.ledger.record(
                    predictions,
                    generated_at=context.now,
                    model_id=str(predictions["model_id"].iloc[0]),
                )
                self._events.append(
                    {
                        "timestamp": context.now,
                        "event": "predictions_generated",
                        "rows": len(predictions),
                    }
                )
            self.clock.advance()

        final_context = self._context()
        scored = self.ledger.score_available(self.source, as_of=final_context.now)
        if not scored.empty:
            self.strategy.observe(final_context, scored)
            self._events.append(
                {
                    "timestamp": final_context.now,
                    "event": "labels_scored",
                    "rows": len(scored),
                }
            )
        return pd.DataFrame(self._events)
