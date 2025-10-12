"""Resumable monthly processing for raw TLC Yellow Taxi files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from taxi_mlops.ingestion.inventory import month_sequence
from taxi_mlops.ingestion.yellow_taxi import (
    DataContractError,
    build_zone_hour_dataset,
    read_trip_file,
)


def _month_boundaries(month: str, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    period = pd.Period(month, freq="M")
    start = period.start_time.tz_localize(timezone)
    end = (period + 1).start_time.tz_localize(timezone)
    return start, end


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def process_month_range(
    *,
    raw_root: str | Path,
    zones_path: str | Path,
    output_root: str | Path,
    report_root: str | Path,
    start_month: str,
    end_month: str,
    timezone: str = "America/New_York",
    force: bool = False,
    allow_missing: bool = False,
) -> dict[str, object]:
    """Process an inclusive month range and safely resume completed outputs."""

    raw_root = Path(raw_root)
    zones_path = Path(zones_path)
    output_root = Path(output_root)
    report_root = Path(report_root)
    months = month_sequence(start_month, end_month)
    raw_paths = {
        month: raw_root / month[:4] / f"yellow_tripdata_{month}.parquet" for month in months
    }
    missing_months = [month for month, path in raw_paths.items() if not path.is_file()]
    if missing_months and not allow_missing:
        preview = ", ".join(missing_months[:6])
        suffix = " ..." if len(missing_months) > 6 else ""
        raise DataContractError(
            f"{len(missing_months)} raw month(s) are missing: {preview}{suffix}"
        )

    zones = pd.read_csv(zones_path)
    processed: list[dict[str, object]] = []
    skipped_existing: list[str] = []
    for month in months:
        raw_path = raw_paths[month]
        if not raw_path.is_file():
            continue
        output_path = output_root / month[:4] / f"zone_hour_{month}.parquet"
        report_path = report_root / month[:4] / f"zone_hour_{month}.json"
        if not force and output_path.is_file() and report_path.is_file():
            skipped_existing.append(month)
            continue

        start, end = _month_boundaries(month, timezone)
        result = build_zone_hour_dataset(
            read_trip_file(raw_path),
            zones,
            start=start,
            end=end,
            source_file=raw_path.name,
            timezone=timezone,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".partial")
        result.data.to_parquet(temporary_output, index=False)
        temporary_output.replace(output_path)
        _atomic_write_json(report_path, result.report.to_dict())
        processed.append(
            {
                "month": month,
                "raw_path": str(raw_path),
                "output_path": str(output_path),
                "report_path": str(report_path),
                "zone_hour_rows": len(result.data),
                "accepted_pickups": int(result.data["pickup_count"].sum()),
                "canonical_hours": result.report.expected_hours,
            }
        )

    return {
        "start_month": start_month,
        "end_month": end_month,
        "requested_month_count": len(months),
        "processed_month_count": len(processed),
        "skipped_existing_month_count": len(skipped_existing),
        "missing_month_count": len(missing_months),
        "missing_months": missing_months,
        "skipped_existing_months": skipped_existing,
        "processed": processed,
    }

