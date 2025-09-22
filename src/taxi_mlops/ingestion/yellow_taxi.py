"""Build the canonical Yellow Taxi zone-hour dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


class DataContractError(ValueError):
    """Raised when source data cannot satisfy the canonical dataset contract."""


@dataclass(frozen=True)
class QualityReport:
    source_rows: int
    rows_in_requested_interval: int
    valid_rows: int
    invalid_timestamp_rows: int
    invalid_zone_rows: int
    invalid_zone_id_counts: dict[str, int]
    out_of_interval_rows: int
    rows_before_start: int
    rows_at_or_after_end: int
    source_min_timestamp_utc: str | None
    source_max_timestamp_utc: str | None
    expected_hours: int
    observed_source_hours: int
    missing_source_hours: tuple[str, ...]
    ambiguous_dst_policy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BuildResult:
    data: pd.DataFrame
    report: QualityReport


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise DataContractError(f"{source} is missing required columns: {names}")


def _as_utc_boundary(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataContractError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def _parse_local_pickup_times(values: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is not None:
        return parsed.dt.tz_convert("UTC")

    # TLC timestamps are local wall-clock values and contain no fold marker.
    # Choosing standard time is deterministic; DST hours remain separately
    # reportable and this choice is part of dataset version v1.
    localized = parsed.dt.tz_localize(
        timezone,
        ambiguous=False,
        nonexistent="shift_forward",
    )
    return localized.dt.tz_convert("UTC")


def build_zone_hour_dataset(
    trips: pd.DataFrame,
    zones: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    source_file: str,
    timezone: str = "America/New_York",
    timestamp_column: str = "tpep_pickup_datetime",
    zone_column: str = "PULocationID",
    zone_lookup_id_column: str = "LocationID",
    excluded_zone_ids: tuple[int, ...] = (264, 265),
    dataset_version: str = "zone-hour-v1",
    require_complete_source_hours: bool = True,
) -> BuildResult:
    """Aggregate trips and fill a validated, complete zone-hour grid.

    Start and end are absolute instants; end is exclusive. A source hour is
    considered observed when at least one raw trip with a valid timestamp falls
    in it, even if that trip later fails the zone validation.
    """

    _require_columns(trips, {timestamp_column, zone_column}, "trip data")
    _require_columns(zones, {zone_lookup_id_column}, "zone lookup")

    start_utc = _as_utc_boundary(start, "start")
    end_utc = _as_utc_boundary(end, "end")
    if end_utc <= start_utc:
        raise DataContractError("end must be later than start")
    if start_utc.minute or start_utc.second or start_utc.microsecond:
        raise DataContractError("start must be aligned to an hour")
    if end_utc.minute or end_utc.second or end_utc.microsecond:
        raise DataContractError("end must be aligned to an hour")

    valid_zone_series = pd.to_numeric(zones[zone_lookup_id_column], errors="coerce").dropna()
    valid_zone_ids = sorted(
        set(valid_zone_series.astype("int64")).difference(excluded_zone_ids)
    )
    if not valid_zone_ids:
        raise DataContractError("zone lookup contains no eligible taxi zones")

    work = trips[[timestamp_column, zone_column]].copy()
    work["pickup_timestamp_utc"] = _parse_local_pickup_times(work[timestamp_column], timezone)
    invalid_timestamp = work["pickup_timestamp_utc"].isna()
    valid_timestamps = work.loc[~invalid_timestamp, "pickup_timestamp_utc"]
    in_interval = work["pickup_timestamp_utc"].ge(start_utc) & work[
        "pickup_timestamp_utc"
    ].lt(end_utc)

    interval = work.loc[in_interval].copy()
    interval["pickup_hour"] = interval["pickup_timestamp_utc"].dt.floor("h")
    interval["pickup_zone_id"] = pd.to_numeric(interval[zone_column], errors="coerce")

    expected_hours = pd.date_range(start=start_utc, end=end_utc, freq="h", inclusive="left")
    observed_hours = pd.DatetimeIndex(interval["pickup_hour"].dropna().unique()).sort_values()
    missing_hours = expected_hours.difference(observed_hours)
    if require_complete_source_hours and not missing_hours.empty:
        preview = ", ".join(str(item) for item in missing_hours[:5])
        suffix = " ..." if len(missing_hours) > 5 else ""
        raise DataContractError(
            f"raw input is missing {len(missing_hours)} expected source hour(s): "
            f"{preview}{suffix}"
        )

    valid_zone_mask = interval["pickup_zone_id"].isin(valid_zone_ids)
    invalid_zone_counts = interval.loc[~valid_zone_mask, "pickup_zone_id"].value_counts(
        dropna=False
    )

    def _zone_count_key(value: object) -> str:
        if pd.isna(value):
            return "<missing>"
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)

    valid = interval.loc[valid_zone_mask, ["pickup_hour", "pickup_zone_id"]].copy()
    valid["pickup_zone_id"] = valid["pickup_zone_id"].astype("int64")

    counts = (
        valid.groupby(["pickup_hour", "pickup_zone_id"], observed=True)
        .size()
        .rename("pickup_count")
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [expected_hours, valid_zone_ids], names=["pickup_hour", "pickup_zone_id"]
    ).to_frame(index=False)
    data = grid.merge(counts, on=["pickup_hour", "pickup_zone_id"], how="left", validate="1:1")
    data["had_pickups"] = data["pickup_count"].notna()
    data["pickup_count"] = data["pickup_count"].fillna(0).astype("int64")
    data["source_hour_observed"] = data["pickup_hour"].isin(observed_hours)
    data["source_file"] = source_file
    data["dataset_version"] = dataset_version

    report = QualityReport(
        source_rows=len(work),
        rows_in_requested_interval=len(interval),
        valid_rows=len(valid),
        invalid_timestamp_rows=int(invalid_timestamp.sum()),
        invalid_zone_rows=int((~valid_zone_mask).sum()),
        invalid_zone_id_counts={
            _zone_count_key(zone_id): int(count)
            for zone_id, count in invalid_zone_counts.items()
        },
        out_of_interval_rows=int((~invalid_timestamp & ~in_interval).sum()),
        rows_before_start=int(
            (~invalid_timestamp & work["pickup_timestamp_utc"].lt(start_utc)).sum()
        ),
        rows_at_or_after_end=int(
            (~invalid_timestamp & work["pickup_timestamp_utc"].ge(end_utc)).sum()
        ),
        source_min_timestamp_utc=(
            str(valid_timestamps.min()) if not valid_timestamps.empty else None
        ),
        source_max_timestamp_utc=(
            str(valid_timestamps.max()) if not valid_timestamps.empty else None
        ),
        expected_hours=len(expected_hours),
        observed_source_hours=len(observed_hours),
        missing_source_hours=tuple(str(item) for item in missing_hours),
        ambiguous_dst_policy="assign_standard_time; shift_nonexistent_forward",
    )
    return BuildResult(data=data, report=report)


def read_trip_file(
    path: str | Path,
    *,
    columns: tuple[str, ...] = ("tpep_pickup_datetime", "PULocationID"),
) -> pd.DataFrame:
    """Read only columns required by the initial aggregation pipeline."""

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=list(columns))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=list(columns))
    raise DataContractError(f"unsupported trip file type: {path.suffix}")
