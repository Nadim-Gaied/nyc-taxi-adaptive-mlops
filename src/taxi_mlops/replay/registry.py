"""Metadata registry for models created during historical replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def _utc(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    trained_at: str
    training_start: str
    training_end: str
    history_strategy: str
    feature_set: str
    artifact_path: str

    def validate(self) -> None:
        trained_at = _utc(self.trained_at, "trained_at")
        training_start = _utc(self.training_start, "training_start")
        training_end = _utc(self.training_end, "training_end")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if training_end <= training_start:
            raise ValueError("model training end must be later than start")
        if training_end > trained_at:
            raise ValueError("a model cannot finish training after it is registered")
        if not Path(self.artifact_path).name:
            raise ValueError("artifact_path must identify a model artifact")


@dataclass(frozen=True)
class PromotionRecord:
    promoted_at: str
    model_id: str
    previous_model_id: str | None
    reason: str


class ModelRegistry:
    """Retain all valid models and enforce temporal eligibility for reuse."""

    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}
        self._promotions: list[PromotionRecord] = []
        self._active_model_id: str | None = None

    @property
    def active_model_id(self) -> str | None:
        return self._active_model_id

    def register(self, record: ModelRecord) -> None:
        record.validate()
        if record.model_id in self._models:
            raise ValueError(f"model_id is already registered: {record.model_id}")
        self._models[record.model_id] = record

    def eligible_for_evaluation(
        self,
        *,
        evaluation_start: str | pd.Timestamp,
        as_of: str | pd.Timestamp,
    ) -> list[ModelRecord]:
        """Exclude models trained on any part of the evaluation window."""

        window_start = _utc(evaluation_start, "evaluation_start")
        cutoff = _utc(as_of, "as_of")
        if window_start >= cutoff:
            raise ValueError("evaluation window must begin before as_of")
        return sorted(
            (
                record
                for record in self._models.values()
                if _utc(record.trained_at, "trained_at") <= window_start
            ),
            key=lambda record: (record.trained_at, record.model_id),
        )

    def promote(
        self,
        model_id: str,
        *,
        as_of: str | pd.Timestamp,
        reason: str,
        cooldown_weeks: int,
    ) -> PromotionRecord:
        if model_id not in self._models:
            raise ValueError(f"cannot promote unregistered model: {model_id}")
        promoted_at = _utc(as_of, "as_of")
        record = self._models[model_id]
        if _utc(record.trained_at, "trained_at") > promoted_at:
            raise ValueError("cannot promote a model before it has been trained")
        if cooldown_weeks < 0:
            raise ValueError("cooldown_weeks cannot be negative")
        if self._promotions:
            previous_time = _utc(self._promotions[-1].promoted_at, "promoted_at")
            eligible_at = previous_time + pd.Timedelta(weeks=cooldown_weeks)
            if promoted_at < eligible_at:
                raise ValueError("model promotion is blocked by the switching cooldown")
        promotion = PromotionRecord(
            promoted_at=str(promoted_at),
            model_id=model_id,
            previous_model_id=self._active_model_id,
            reason=reason,
        )
        self._promotions.append(promotion)
        self._active_model_id = model_id
        return promotion

    def models(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self._models.values()]

    def records(self) -> list[ModelRecord]:
        return list(self._models.values())

    def promotions(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self._promotions]

    def write_manifest(self, path: str | Path) -> None:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "active_model_id": self._active_model_id,
                    "models": self.models(),
                    "promotions": self.promotions(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_manifest(cls, path: str | Path) -> ModelRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported model-registry manifest version")
        registry = cls()
        for raw_record in payload.get("models", []):
            registry.register(ModelRecord(**raw_record))
        previous_time: pd.Timestamp | None = None
        for raw_promotion in payload.get("promotions", []):
            promotion = PromotionRecord(**raw_promotion)
            if promotion.model_id not in registry._models:
                raise ValueError("registry promotion refers to an unknown model")
            promoted_at = _utc(promotion.promoted_at, "promoted_at")
            if previous_time is not None and promoted_at < previous_time:
                raise ValueError("registry promotions are not chronological")
            registry._promotions.append(promotion)
            previous_time = promoted_at
        active_model_id = payload.get("active_model_id")
        if active_model_id is not None and active_model_id not in registry._models:
            raise ValueError("registry active model is not registered")
        if registry._promotions and registry._promotions[-1].model_id != active_model_id:
            raise ValueError("registry active model differs from its last promotion")
        registry._active_model_id = active_model_id
        return registry
