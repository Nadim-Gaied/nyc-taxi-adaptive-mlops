"""Reconstruct the pre-registered model pool before locked replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from taxi_mlops.models.training_history import _history_intervals
from taxi_mlops.replay.modeling import ReplayModelStore


def seed_replay_registry(
    *,
    source,
    model_store: ReplayModelStore,
    origins: Sequence[str | pd.Timestamp],
    histories: Sequence[str],
    feature_columns: Sequence[str],
    feature_set: str,
    timezone: str,
    model_parameters: Mapping[str, object] | None,
    manifest_path: str | Path,
) -> pd.DataFrame:
    """Train the fixed pre-replay pool, checkpointing after every model."""

    manifest_path = Path(manifest_path)
    existing = {record.model_id for record in model_store.registry.records()}
    rows: list[dict[str, object]] = []
    for raw_origin in origins:
        origin = pd.Timestamp(raw_origin)
        if origin.tzinfo is None:
            raise ValueError("seed origins must include an explicit UTC offset")
        origin = origin.tz_convert("UTC")
        for history in histories:
            model_id = f"seed__{history}__{origin.strftime('%Y%m%dT%H%M%SZ')}"
            if model_id in existing:
                record = next(
                    record
                    for record in model_store.registry.records()
                    if record.model_id == model_id
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "origin": str(origin),
                        "history_strategy": history,
                        "status": "resumed",
                        "training_rows": None,
                        "training_seconds": None,
                        "artifact_path": record.artifact_path,
                    }
                )
                continue
            intervals = _history_intervals(history, origin, source.data_start, timezone)
            training_data = source.observations_as_of(
                origin, start=min(start for start, _ in intervals)
            )
            outcome = model_store.train(
                training_data,
                model_id=model_id,
                as_of=origin,
                registered_at=origin,
                history_strategy=history,
                feature_set=feature_set,
                feature_columns=feature_columns,
                timezone=timezone,
                model_parameters=model_parameters,
            )
            model_store.registry.write_manifest(manifest_path)
            existing.add(model_id)
            rows.append(
                {
                    "model_id": model_id,
                    "origin": str(origin),
                    "history_strategy": history,
                    "status": "trained",
                    "training_rows": outcome.training_rows,
                    "training_seconds": outcome.training_seconds,
                    "artifact_path": outcome.record.artifact_path,
                }
            )
            print(f"Seeded {history} at {origin}", flush=True)
    return pd.DataFrame(rows)
