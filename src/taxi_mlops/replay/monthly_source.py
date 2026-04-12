"""File-backed replay source for month-partitioned feature tables."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from taxi_mlops.ingestion.inventory import month_sequence
from taxi_mlops.replay.source import TARGET_COLUMNS, _utc


class MonthlyReplayDataSource:
    """Expose the ReplayDataSource contract without loading the full replay."""

    def __init__(
        self,
        *,
        feature_root: str | Path,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        timezone: str = "America/New_York",
        cache_months: int = 2,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.timezone = timezone
        self._start = _utc(start, "start")
        self._end = _utc(end, "end")
        if self._end <= self._start:
            raise ValueError("monthly replay source end must be later than start")
        if cache_months < 1:
            raise ValueError("cache_months must be positive")
        self.cache_months = cache_months
        self._cache: OrderedDict[str, tuple[pd.DataFrame, pd.DatetimeIndex]] = OrderedDict()

    @property
    def data_start(self) -> pd.Timestamp:
        return self._start

    def _path(self, month: str) -> Path:
        return self.feature_root / month[:4] / f"features_{month}.parquet"

    def _read_month(self, month: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        if month in self._cache:
            partition = self._cache.pop(month)
            self._cache[month] = partition
            return partition
        path = self._path(month)
        if not path.is_file():
            raise FileNotFoundError(f"missing replay feature month: {path}")
        frame = pd.read_parquet(path)
        frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"], utc=True, errors="raise")
        frame = frame.sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(drop=True)
        partition = (frame, pd.DatetimeIndex(frame["pickup_hour"]))
        self._cache[month] = partition
        while len(self._cache) > self.cache_months:
            self._cache.popitem(last=False)
        return partition

    def _months(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        if end <= start:
            return []
        local_start = start.tz_convert(self.timezone)
        local_last = (end - pd.Timedelta(nanoseconds=1)).tz_convert(self.timezone)
        return month_sequence(local_start.strftime("%Y-%m"), local_last.strftime("%Y-%m"))

    def _range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        bounded_start = max(start, self._start)
        bounded_end = min(end, self._end)
        months = self._months(bounded_start, bounded_end)
        if not months:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for month in months:
            frame, hours = self._read_month(month)
            left = int(hours.searchsorted(bounded_start, side="left"))
            right = int(hours.searchsorted(bounded_end, side="left"))
            frames.append(frame.iloc[left:right])
        return pd.concat(frames, ignore_index=True).copy()

    def observations_as_of(
        self,
        as_of: str | pd.Timestamp,
        *,
        start: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        cutoff = _utc(as_of, "as_of")
        start_utc = self._start if start is None else _utc(start, "start")
        return self._range(start_utc, cutoff)

    def labels_between(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        as_of: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start_utc = _utc(start, "start")
        end_utc = _utc(end, "end")
        cutoff = _utc(as_of, "as_of")
        if end_utc > cutoff:
            raise ValueError("requested labels include hours not yet released")
        if end_utc <= start_utc:
            raise ValueError("label interval end must be later than start")
        return self._range(start_utc, end_utc)[TARGET_COLUMNS]

    def prediction_features(
        self,
        target_hour: str | pd.Timestamp,
        *,
        as_of: str | pd.Timestamp,
    ) -> pd.DataFrame:
        target = _utc(target_hour, "target_hour")
        cutoff = _utc(as_of, "as_of")
        if target != cutoff:
            raise ValueError("one-hour replay predictions must target the current clock hour")
        end = target + pd.Timedelta(hours=1)
        rows = self._range(target, end)
        return rows.loc[rows["pickup_hour"].eq(target)].drop(columns="pickup_count")
