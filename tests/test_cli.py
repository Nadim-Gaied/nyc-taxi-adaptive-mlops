from __future__ import annotations

import json
import sys

import pandas as pd

from taxi_mlops.cli import main


def test_cli_builds_parquet_and_quality_report(tmp_path, monkeypatch) -> None:
    trips_path = tmp_path / "trips.parquet"
    zones_path = tmp_path / "zones.csv"
    output_path = tmp_path / "zone_hour.parquet"
    report_path = tmp_path / "quality.json"

    pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2019-01-01 00:10:00",
                "2019-01-01 01:10:00",
            ],
            "PULocationID": [1, 1],
        }
    ).to_parquet(trips_path, index=False)
    pd.DataFrame({"LocationID": [1, 2]}).to_csv(zones_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "taxi-mlops",
            "build-dataset",
            "--trips",
            str(trips_path),
            "--zones",
            str(zones_path),
            "--start",
            "2019-01-01T00:00:00-05:00",
            "--end",
            "2019-01-01T02:00:00-05:00",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )
    main()

    output = pd.read_parquet(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(output) == 4
    assert output["pickup_count"].sum() == 2
    assert report["expected_hours"] == 2
    assert report["missing_source_hours"] == []

