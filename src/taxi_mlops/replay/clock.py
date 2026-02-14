"""Virtual production clock for chronological historical replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _as_utc(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def shift_local_calendar_days(
    value: str | pd.Timestamp, *, days: int, timezone: str
) -> pd.Timestamp:
    """Shift by local calendar days using the project's DST resolution policy."""

    timestamp = _as_utc(value, "value")
    local_naive = timestamp.tz_convert(timezone).tz_localize(None)
    shifted_naive = local_naive + pd.DateOffset(days=days)
    return shifted_naive.tz_localize(
        timezone,
        ambiguous=False,
        nonexistent="shift_forward",
    ).tz_convert("UTC")


@dataclass
class ReplayClock:
    """Advance an hourly clock while anchoring decisions to local calendar days."""

    start: str | pd.Timestamp
    end: str | pd.Timestamp
    timezone: str = "America/New_York"
    decision_cadence_days: int = 7
    now: pd.Timestamp = field(init=False)
    _end: pd.Timestamp = field(init=False, repr=False)
    _next_decision: pd.Timestamp = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.now = _as_utc(self.start, "start")
        self._end = _as_utc(self.end, "end")
        if self._end <= self.now:
            raise ValueError("replay end must be later than start")
        if self.decision_cadence_days <= 0:
            raise ValueError("decision cadence must be positive")
        self._next_decision = shift_local_calendar_days(
            self.now,
            days=self.decision_cadence_days,
            timezone=self.timezone,
        )

    @property
    def finished(self) -> bool:
        return self.now >= self._end

    @property
    def end_utc(self) -> pd.Timestamp:
        return self._end

    def decision_due(self) -> bool:
        return self.now >= self._next_decision and not self.finished

    def mark_decision(self) -> None:
        if not self.decision_due():
            raise ValueError("no lifecycle decision is due at the current replay time")
        self._next_decision = shift_local_calendar_days(
            self._next_decision,
            days=self.decision_cadence_days,
            timezone=self.timezone,
        )

    def advance(self) -> pd.Timestamp:
        if self.finished:
            raise StopIteration("replay clock has reached its end")
        self.now = min(self.now + pd.Timedelta(hours=1), self._end)
        return self.now
