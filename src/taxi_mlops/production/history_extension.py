"""Extend dashboard history with models available at each historical hour."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from taxi_mlops.production.backfill import ProductionBackfillRunner
from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.models import ResolvedModel
from taxi_mlops.production.state import ProductionStateStore


@dataclass(frozen=True)
class HistoricalEpoch:
    start: pd.Timestamp
    end: pd.Timestamp
    model_id: str
    model_path: Path


class _Resolver:
    def __init__(self, model: ResolvedModel) -> None:
        self.model = model

    def resolve(self) -> ResolvedModel:
        return self.model


def _unobservable_fall_back_hours(
    start: pd.Timestamp, end: pd.Timestamp, timezone: str
) -> set[pd.Timestamp]:
    """TLC wall-clock timestamps cannot identify the first repeated local hour."""
    hours = pd.date_range(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), freq="h")
    wall_clock = hours.tz_convert(timezone).tz_localize(None)
    return set(hours[wall_clock.duplicated(keep="last")])


def plan_historical_epochs(
    registry_path: str | Path,
    *,
    start: str,
    end: str,
) -> list[HistoricalEpoch]:
    """Split a UTC interval at every causally available model registration."""
    begin = canonical_utc(start, "start")
    finish = canonical_utc(end, "end")
    if begin != begin.floor("h") or finish != finish.floor("h") or finish <= begin:
        raise ValueError("history boundaries must be increasing whole UTC hours")
    registry_path = Path(registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    models = sorted(
        payload["models"], key=lambda item: canonical_utc(item["trained_at"], "trained_at")
    )
    available = [
        item for item in models if canonical_utc(item["trained_at"], "trained_at") <= begin
    ]
    if not available:
        raise ValueError("no registered model was available at the first requested hour")
    changes = [
        item for item in models if begin < canonical_utc(item["trained_at"], "trained_at") < finish
    ]
    epochs = []
    for item, next_item in zip([available[-1], *changes], [*changes, None], strict=True):
        epoch_start = max(begin, canonical_utc(item["trained_at"], "trained_at"))
        epoch_end = (
            canonical_utc(next_item["trained_at"], "trained_at")
            if next_item is not None
            else finish
        )
        model_path = Path(item["artifact_path"])
        if not model_path.is_absolute():
            model_path = registry_path.parent / model_path
        epochs.append(
            HistoricalEpoch(epoch_start, epoch_end, item["model_id"], model_path.resolve())
        )
    return epochs


def extend_historical_scores(
    *,
    config: ProductionConfig,
    state: ProductionStateStore,
    registry_path: str | Path,
    start: str,
    end: str,
    maximum_hours: int = 744,
) -> list[dict[str, object]]:
    """Predict and score each hour while preserving its model lineage."""
    if maximum_hours < 1 or maximum_hours > 744:
        raise ValueError("maximum_hours must be between 1 and 744")
    epochs = plan_historical_epochs(registry_path, start=start, end=end)
    skipped = _unobservable_fall_back_hours(epochs[0].start, epochs[-1].end, config.timezone)
    results = []
    for epoch in epochs:
        if not epoch.model_path.is_file():
            raise FileNotFoundError(epoch.model_path)
        metadata = json.loads(epoch.model_path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["model_id"] != epoch.model_id:
            raise ValueError(f"metadata does not match {epoch.model_id}")
        booster = lgb.Booster(model_file=str(epoch.model_path))
        if list(metadata["feature_columns"]) != booster.feature_name():
            raise ValueError(f"feature schema does not match {epoch.model_id}")
        model = ResolvedModel(
            name="nyc-taxi-frozen-replay",
            version=epoch.model_id,
            alias="historical",
            run_id=epoch.model_id,
            source_uri=epoch.model_path.as_uri(),
            booster=booster,
        )
        runner = ProductionBackfillRunner(
            config=config, state_store=state, model_resolver=_Resolver(model)
        )
        cursor = epoch.start
        while cursor < epoch.end:
            if cursor in skipped:
                results.append(
                    {
                        "skipped_hour": cursor.isoformat(),
                        "reason": "TLC source has no fold marker for the first repeated local hour",
                    }
                )
                cursor += pd.Timedelta(hours=1)
                continue
            chunk_end = min(cursor + pd.Timedelta(hours=maximum_hours), epoch.end)
            next_skip = min((hour for hour in skipped if cursor < hour < chunk_end), default=None)
            if next_skip is not None:
                chunk_end = next_skip
            results.append(
                runner.run(start=cursor, end=chunk_end, maximum_hours=maximum_hours).to_dict()
            )
            cursor = chunk_end
    return results
