"""Train, persist, load, and score LightGBM models during replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import pandas as pd

from taxi_mlops.models.training_history import _history_intervals, _select_intervals
from taxi_mlops.models.window_experiment import DEFAULT_MODEL_PARAMETERS
from taxi_mlops.replay.registry import ModelRecord, ModelRegistry


@dataclass(frozen=True)
class TrainingOutcome:
    record: ModelRecord
    training_rows: int
    training_seconds: float


@dataclass
class _LoadedModel:
    booster: lgb.Booster
    feature_columns: list[str]
    zone_categories: list[int]


class ReplayModelStore:
    """Artifact store coupled to one strategy's independent model registry."""

    def __init__(self, artifact_dir: str | Path, registry: ModelRegistry) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self._loaded: dict[str, _LoadedModel] = {}

    def train(
        self,
        feature_table: pd.DataFrame,
        *,
        model_id: str,
        as_of: str | pd.Timestamp,
        registered_at: str | pd.Timestamp | None = None,
        history_strategy: str,
        feature_set: str,
        feature_columns: Sequence[str],
        timezone: str,
        model_parameters: Mapping[str, object] | None = None,
    ) -> TrainingOutcome:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError("training cutoff must include an explicit UTC offset")
        cutoff = cutoff.tz_convert("UTC")
        registration_time = pd.Timestamp(registered_at or cutoff)
        if registration_time.tzinfo is None:
            raise ValueError("registration time must include an explicit UTC offset")
        registration_time = registration_time.tz_convert("UTC")
        if registration_time < cutoff:
            raise ValueError("a model cannot be registered before its training cutoff")
        data = feature_table.copy()
        data["pickup_hour"] = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
        if data["pickup_hour"].ge(cutoff).any():
            raise ValueError("training input contains observations unavailable at cutoff")
        columns = list(feature_columns)
        required = {"pickup_hour", "pickup_zone_id", "pickup_count", *columns}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"training data is missing columns: {', '.join(sorted(missing))}")
        data_start = pd.Timestamp(data["pickup_hour"].min())
        intervals = _history_intervals(history_strategy, cutoff, data_start, timezone)
        train = _select_intervals(data, intervals)
        if train.empty:
            raise ValueError(f"no training rows for {history_strategy} at {cutoff}")

        zone_categories = sorted(int(value) for value in train["pickup_zone_id"].unique())
        x_train = train[columns].copy()
        x_train["pickup_zone_id"] = pd.Categorical(
            x_train["pickup_zone_id"], categories=zone_categories
        )
        parameters = {**DEFAULT_MODEL_PARAMETERS, **(model_parameters or {})}
        estimator = lgb.LGBMRegressor(**parameters)
        started = perf_counter()
        estimator.fit(
            x_train,
            train["pickup_count"],
            categorical_feature=["pickup_zone_id"],
        )
        training_seconds = perf_counter() - started

        model_path = self.artifact_dir / f"{model_id}.txt"
        metadata_path = self.artifact_dir / f"{model_id}.json"
        estimator.booster_.save_model(str(model_path))
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "feature_columns": columns,
                    "zone_categories": zone_categories,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        record = ModelRecord(
            model_id=model_id,
            trained_at=str(registration_time),
            training_start=str(min(start for start, _ in intervals)),
            training_end=str(cutoff),
            history_strategy=history_strategy,
            feature_set=feature_set,
            artifact_path=str(model_path),
        )
        self.registry.register(record)
        self._loaded[model_id] = _LoadedModel(
            booster=estimator.booster_,
            feature_columns=columns,
            zone_categories=zone_categories,
        )
        return TrainingOutcome(
            record=record,
            training_rows=int(len(train)),
            training_seconds=float(training_seconds),
        )

    def load(self, record: ModelRecord) -> None:
        metadata_path = Path(record.artifact_path).with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["model_id"] != record.model_id:
            raise ValueError("model metadata does not match registry record")
        self._loaded[record.model_id] = _LoadedModel(
            booster=lgb.Booster(model_file=record.artifact_path),
            feature_columns=list(metadata["feature_columns"]),
            zone_categories=[int(value) for value in metadata["zone_categories"]],
        )

    def predict(self, model_id: str, features: pd.DataFrame) -> tuple[pd.Series, float]:
        if model_id not in self._loaded:
            records = [record for record in self.registry.records() if record.model_id == model_id]
            if not records:
                raise ValueError(f"model is not registered: {model_id}")
            self.load(records[0])
        loaded = self._loaded[model_id]
        missing = set(loaded.feature_columns).difference(features.columns)
        if missing:
            raise ValueError(
                f"prediction features are missing columns: {', '.join(sorted(missing))}"
            )
        matrix = features[loaded.feature_columns].copy()
        matrix["pickup_zone_id"] = pd.Categorical(
            matrix["pickup_zone_id"], categories=loaded.zone_categories
        )
        started = perf_counter()
        prediction = loaded.booster.predict(matrix)
        inference_seconds = perf_counter() - started
        return pd.Series(prediction, index=features.index).clip(lower=0.0), float(inference_seconds)
