from __future__ import annotations

import json

import pandas as pd
import pytest

from taxi_mlops.ingestion import process_month_range
from taxi_mlops.ingestion.yellow_taxi import DataContractError


def _write_complete_raw_month(root, month: str) -> None:
    period = pd.Period(month, freq="M")
    hours = pd.date_range(period.start_time, (period + 1).start_time, freq="h", inclusive="left")
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": hours + pd.Timedelta(minutes=10),
            "PULocationID": 1,
        }
    )
    year_dir = root / month[:4]
    year_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(year_dir / f"yellow_tripdata_{month}.parquet", index=False)


def test_process_month_range_writes_and_resumes_outputs(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    report_root = tmp_path / "reports"
    zones_path = tmp_path / "zones.csv"
    pd.DataFrame({"LocationID": [1, 2]}).to_csv(zones_path, index=False)
    _write_complete_raw_month(raw_root, "2019-01")

    first = process_month_range(
        raw_root=raw_root,
        zones_path=zones_path,
        output_root=output_root,
        report_root=report_root,
        start_month="2019-01",
        end_month="2019-01",
    )
    second = process_month_range(
        raw_root=raw_root,
        zones_path=zones_path,
        output_root=output_root,
        report_root=report_root,
        start_month="2019-01",
        end_month="2019-01",
    )

    output = pd.read_parquet(output_root / "2019" / "zone_hour_2019-01.parquet")
    report = json.loads(
        (report_root / "2019" / "zone_hour_2019-01.json").read_text(encoding="utf-8")
    )
    assert first["processed_month_count"] == 1
    assert second["skipped_existing_months"] == ["2019-01"]
    assert len(output) == 744 * 2
    assert report["expected_hours"] == 744


def test_process_month_range_fails_before_work_when_month_is_missing(tmp_path) -> None:
    zones_path = tmp_path / "zones.csv"
    pd.DataFrame({"LocationID": [1]}).to_csv(zones_path, index=False)

    with pytest.raises(DataContractError, match="1 raw month"):
        process_month_range(
            raw_root=tmp_path / "raw",
            zones_path=zones_path,
            output_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_month="2019-01",
            end_month="2019-01",
        )
