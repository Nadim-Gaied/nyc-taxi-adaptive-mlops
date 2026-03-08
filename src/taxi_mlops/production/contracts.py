"""Validation and content identities for production data contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from taxi_mlops.replay.source import KEY_COLUMNS


def canonical_utc(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def validate_prediction_features(
    features: pd.DataFrame,
    *,
    target_hour: str | pd.Timestamp,
    feature_columns: Sequence[str],
    expected_zones: int | None,
) -> pd.DataFrame:
    target = canonical_utc(target_hour, "target_hour")
    required = {*KEY_COLUMNS, *feature_columns}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"prediction features are missing: {', '.join(sorted(missing))}")
    if "pickup_count" in features:
        raise ValueError("prediction features contain the hidden target")
    frame = features.copy()
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"], utc=True, errors="raise")
    if frame.empty:
        raise ValueError("prediction feature batch is empty")
    if not frame["pickup_hour"].eq(target).all():
        raise ValueError("prediction feature batch contains the wrong target hour")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("prediction feature batch contains duplicate zone-hour keys")
    if expected_zones is not None and len(frame) != expected_zones:
        raise ValueError(
            f"prediction feature batch has {len(frame)} zones; expected {expected_zones}"
        )
    return frame.sort_values(KEY_COLUMNS).reset_index(drop=True)


def validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected_keys: pd.DataFrame,
) -> pd.DataFrame:
    required = {*KEY_COLUMNS, "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing: {', '.join(sorted(missing))}")
    frame = predictions[[*KEY_COLUMNS, "prediction"]].copy()
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"], utc=True, errors="raise")
    frame = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
    expected = expected_keys[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not frame[KEY_COLUMNS].equals(expected):
        raise ValueError("predictions do not cover the complete expected target grid")
    values = frame["prediction"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    if (values < 0).any():
        raise ValueError("predictions contain negative values")
    return frame


def dataframe_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Return a stable hash for an ordered, point-in-time feature batch."""

    selected = frame[list(columns)].copy()
    for name in selected.select_dtypes(include=["category"]).columns:
        selected[name] = selected[name].astype("string")
    row_hashes = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    schema = [(name, str(selected[name].dtype)) for name in selected.columns]
    digest = hashlib.sha256()
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()
