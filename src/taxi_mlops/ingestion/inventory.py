"""Inventory local monthly TLC Parquet files without loading their trip rows."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

YELLOW_FILE_PATTERN = re.compile(r"^yellow_tripdata_(\d{4})-(\d{2})\.parquet$")


def _month_sequence(start_month: str, end_month: str) -> list[str]:
    try:
        start_year, start_number = (int(part) for part in start_month.split("-"))
        end_year, end_number = (int(part) for part in end_month.split("-"))
    except (TypeError, ValueError) as error:
        raise ValueError("months must use YYYY-MM format") from error
    if not 1 <= start_number <= 12 or not 1 <= end_number <= 12:
        raise ValueError("month number must be between 01 and 12")
    start_index = start_year * 12 + start_number - 1
    end_index = end_year * 12 + end_number - 1
    if end_index < start_index:
        raise ValueError("end_month must not be earlier than start_month")
    return [
        f"{index // 12:04d}-{index % 12 + 1:02d}"
        for index in range(start_index, end_index + 1)
    ]


def _column_min_max(
    parquet_file: pq.ParquetFile, column_name: str
) -> tuple[str | None, str | None]:
    if column_name not in parquet_file.schema.names:
        return None, None
    column_index = parquet_file.schema.names.index(column_name)
    minima: list[object] = []
    maxima: list[object] = []
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(column_index)
        statistics = column.statistics
        if statistics is not None and statistics.has_min_max:
            minima.append(statistics.min)
            maxima.append(statistics.max)
    if not minima:
        return None, None
    return str(min(minima)), str(max(maxima))


def inventory_yellow_files(
    root: str | Path,
    *,
    start_month: str,
    end_month: str,
) -> dict[str, object]:
    """Describe present files and gaps across an expected inclusive month range."""

    root = Path(root)
    expected_months = _month_sequence(start_month, end_month)
    discovered: dict[str, list[Path]] = {}
    ignored_parquet_files: list[str] = []
    if root.exists():
        for path in root.rglob("*.parquet"):
            match = YELLOW_FILE_PATTERN.match(path.name)
            if match is None:
                ignored_parquet_files.append(str(path))
                continue
            month = f"{match.group(1)}-{match.group(2)}"
            discovered.setdefault(month, []).append(path)

    files: list[dict[str, object]] = []
    for month in sorted(discovered):
        for path in sorted(discovered[month]):
            parquet_file = pq.ParquetFile(path)
            minimum, maximum = _column_min_max(parquet_file, "tpep_pickup_datetime")
            files.append(
                {
                    "month": month,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "rows": parquet_file.metadata.num_rows,
                    "row_groups": parquet_file.metadata.num_row_groups,
                    "columns": parquet_file.schema.names,
                    "pickup_timestamp_min": minimum,
                    "pickup_timestamp_max": maximum,
                }
            )

    expected_set = set(expected_months)
    present_set = set(discovered)
    duplicate_months = {
        month: [str(path) for path in paths]
        for month, paths in discovered.items()
        if len(paths) > 1
    }
    return {
        "root": str(root),
        "requested_start_month": start_month,
        "requested_end_month": end_month,
        "expected_month_count": len(expected_months),
        "present_month_count": len(expected_set.intersection(present_set)),
        "missing_months": sorted(expected_set.difference(present_set)),
        "months_outside_requested_range": sorted(present_set.difference(expected_set)),
        "duplicate_months": duplicate_months,
        "ignored_parquet_files": sorted(ignored_parquet_files),
        "files": files,
    }
