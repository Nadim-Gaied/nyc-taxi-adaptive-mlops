"""Enforce declared temporal split boundaries when loading processed data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from taxi_mlops.ingestion.inventory import month_sequence
from taxi_mlops.ingestion.yellow_taxi import DataContractError

ALLOWED_PURPOSES = {
    "development": {"eda", "development", "validation", "replay"},
    "policy_validation": {"validation", "replay"},
    "locked_replay": {"replay"},
}


def _read_manifest(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise DataContractError("split manifest must be a YAML mapping")
    return manifest


def split_months(
    manifest_path: str | Path,
    split_name: str,
) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    """Return expected months and absolute boundaries for a named split."""

    manifest = _read_manifest(manifest_path)
    if split_name not in ALLOWED_PURPOSES:
        raise DataContractError(f"unknown split: {split_name}")
    split = manifest.get(split_name)
    if not isinstance(split, dict) or "start" not in split or "end" not in split:
        raise DataContractError(f"split manifest has no valid {split_name} interval")

    start = pd.Timestamp(split["start"])
    end = pd.Timestamp(split["end"])
    if start.tzinfo is None or end.tzinfo is None:
        raise DataContractError(f"{split_name} boundaries must include UTC offsets")
    if end <= start:
        raise DataContractError(f"{split_name} end must be later than its start")
    if (
        start.day != 1
        or end.day != 1
        or any((start.hour, start.minute, start.second, end.hour, end.minute, end.second))
    ):
        raise DataContractError(f"{split_name} boundaries must be aligned to local months")

    final_inclusive_month = (end - pd.Timedelta(days=1)).strftime("%Y-%m")
    months = month_sequence(start.strftime("%Y-%m"), final_inclusive_month)
    return months, start.tz_convert("UTC"), end.tz_convert("UTC")


def load_named_split(
    *,
    monthly_root: str | Path,
    manifest_path: str | Path,
    split_name: str,
    purpose: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a complete named split only when its declared access permits it."""

    if split_name not in ALLOWED_PURPOSES:
        raise DataContractError(f"unknown split: {split_name}")
    if purpose not in ALLOWED_PURPOSES[split_name]:
        allowed = ", ".join(sorted(ALLOWED_PURPOSES[split_name]))
        raise DataContractError(
            f"split {split_name} cannot be loaded for {purpose}; allowed purposes: {allowed}"
        )

    months, start_utc, end_utc = split_months(manifest_path, split_name)
    monthly_root = Path(monthly_root)
    paths = {month: monthly_root / month[:4] / f"zone_hour_{month}.parquet" for month in months}
    missing = [month for month, path in paths.items() if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:6])
        suffix = " ..." if len(missing) > 6 else ""
        raise DataContractError(
            f"processed split {split_name} is missing {len(missing)} month(s): {preview}{suffix}"
        )

    frames = [pd.read_parquet(paths[month], columns=columns) for month in months]
    data = pd.concat(frames, ignore_index=True)
    if "pickup_hour" not in data.columns:
        raise DataContractError("loaded split must include pickup_hour")
    timestamps = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
    outside = timestamps.lt(start_utc) | timestamps.ge(end_utc)
    if outside.any():
        raise DataContractError(
            f"processed split {split_name} contains {int(outside.sum())} out-of-bound row(s)"
        )
    return data.sort_values(["pickup_hour", "pickup_zone_id"]).reset_index(drop=True)
