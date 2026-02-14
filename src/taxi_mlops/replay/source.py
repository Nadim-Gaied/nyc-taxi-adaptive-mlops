"""Future-data gate and append-only prediction ledger."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

KEY_COLUMNS = ["pickup_hour", "pickup_zone_id"]
TARGET_COLUMNS = [*KEY_COLUMNS, "pickup_count"]


def _utc(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


class ReplayDataSource:
    """Own the full table but expose labels only after their hour is complete."""

    def __init__(self, feature_table: pd.DataFrame) -> None:
        required = set(TARGET_COLUMNS)
        missing = required.difference(feature_table.columns)
        if missing:
            raise ValueError(f"replay data is missing columns: {', '.join(sorted(missing))}")
        data = feature_table.copy()
        data["pickup_hour"] = pd.to_datetime(
            data["pickup_hour"], utc=True, errors="raise"
        )
        if data.duplicated(KEY_COLUMNS).any():
            raise ValueError("replay data contains duplicate zone-hour keys")
        self._data = data.sort_values(KEY_COLUMNS).reset_index(drop=True)
        self._hours = pd.DatetimeIndex(self._data["pickup_hour"])

    @property
    def data_start(self) -> pd.Timestamp:
        return pd.Timestamp(self._hours.min())

    def _bounds(
        self, start: pd.Timestamp | None, end: pd.Timestamp
    ) -> tuple[int, int]:
        left = 0 if start is None else int(self._hours.searchsorted(start, side="left"))
        right = int(self._hours.searchsorted(end, side="left"))
        return left, right

    def observations_as_of(
        self,
        as_of: str | pd.Timestamp,
        *,
        start: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Release completed labels; the hour beginning at as_of remains hidden."""

        cutoff = _utc(as_of, "as_of")
        start_utc = _utc(start, "start") if start is not None else None
        left, right = self._bounds(start_utc, cutoff)
        return self._data.iloc[left:right].copy()

    def labels_between(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        as_of: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Release labels only when the complete requested interval is observable."""

        start_utc = _utc(start, "start")
        end_utc = _utc(end, "end")
        cutoff = _utc(as_of, "as_of")
        if end_utc > cutoff:
            raise ValueError("requested labels include hours not yet released")
        if end_utc <= start_utc:
            raise ValueError("label interval end must be later than start")
        left, right = self._bounds(start_utc, end_utc)
        return self._data.iloc[left:right][TARGET_COLUMNS].copy()

    def prediction_features(
        self,
        target_hour: str | pd.Timestamp,
        *,
        as_of: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return target keys and features, never the hidden target value."""

        target = _utc(target_hour, "target_hour")
        cutoff = _utc(as_of, "as_of")
        if target != cutoff:
            raise ValueError("one-hour replay predictions must target the current clock hour")
        left = int(self._hours.searchsorted(target, side="left"))
        right = int(self._hours.searchsorted(target, side="right"))
        rows = self._data.iloc[left:right].copy()
        return rows.drop(columns="pickup_count")


@dataclass
class PredictionLedger:
    """Append-only predictions with generation-time and scoring-time checks."""

    _batches: list[pd.DataFrame] = field(default_factory=list)
    _pending: deque[pd.DataFrame] = field(default_factory=deque)
    _last_target_hour: pd.Timestamp | None = None
    retain_scored: bool = True

    def record(
        self,
        predictions: pd.DataFrame,
        *,
        generated_at: str | pd.Timestamp,
        model_id: str,
    ) -> None:
        required = {*KEY_COLUMNS, "prediction"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(
                f"predictions are missing columns: {', '.join(sorted(missing))}"
            )
        frame = predictions[[*KEY_COLUMNS, "prediction"]].copy()
        frame["pickup_hour"] = pd.to_datetime(
            frame["pickup_hour"], utc=True, errors="raise"
        )
        generated = _utc(generated_at, "generated_at")
        if frame["pickup_hour"].lt(generated).any():
            raise ValueError("predictions cannot be generated after their target hour")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError("prediction batch contains duplicate target keys")
        if (
            self._last_target_hour is not None
            and frame["pickup_hour"].min() <= self._last_target_hour
        ):
            raise ValueError("prediction batches must have strictly increasing target hours")
        frame["model_id"] = model_id
        frame["generated_at"] = generated
        frame["scored"] = False
        frame = frame.reset_index(drop=True)
        if self.retain_scored:
            self._batches.append(frame)
        self._pending.append(frame)
        self._last_target_hour = frame["pickup_hour"].max()

    def score_available(
        self,
        source: ReplayDataSource,
        *,
        as_of: str | pd.Timestamp,
    ) -> pd.DataFrame:
        cutoff = _utc(as_of, "as_of")
        released: list[pd.DataFrame] = []
        while self._pending and self._pending[0]["pickup_hour"].max() < cutoff:
            batch = self._pending.popleft()
            batch["scored"] = True
            released.append(batch.drop(columns="scored"))
        if not released:
            return pd.DataFrame(
                columns=[*TARGET_COLUMNS, "prediction", "model_id", "generated_at"]
            )
        pending = pd.concat(released, ignore_index=True)
        labels = source.labels_between(
            pending["pickup_hour"].min(), cutoff, as_of=cutoff
        )
        scored = pending.merge(
            labels,
            on=KEY_COLUMNS,
            how="left",
            validate="1:1",
        )
        if scored["pickup_count"].isna().any():
            raise ValueError("released target grid is missing labels for predictions")
        return scored[
            [*TARGET_COLUMNS, "prediction", "model_id", "generated_at"]
        ].sort_values(KEY_COLUMNS)

    def frame(self) -> pd.DataFrame:
        if not self._batches:
            return pd.DataFrame(
                columns=[
                    *KEY_COLUMNS,
                    "prediction",
                    "model_id",
                    "generated_at",
                    "scored",
                ]
            )
        return pd.concat(self._batches, ignore_index=True)
