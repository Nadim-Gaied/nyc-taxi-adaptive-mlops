from __future__ import annotations

import pandas as pd

from taxi_mlops.ingestion import inventory_yellow_files


def test_inventory_reports_present_and_missing_months(tmp_path) -> None:
    year_dir = tmp_path / "2019"
    year_dir.mkdir()
    pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2019-01-01 00:00:00", "2019-01-31 23:59:59"]),
            "PULocationID": [1, 2],
        }
    ).to_parquet(year_dir / "yellow_tripdata_2019-01.parquet", index=False)

    inventory = inventory_yellow_files(
        tmp_path,
        start_month="2019-01",
        end_month="2019-03",
    )

    assert inventory["present_month_count"] == 1
    assert inventory["missing_months"] == ["2019-02", "2019-03"]
    assert inventory["files"][0]["rows"] == 2
    assert inventory["files"][0]["pickup_timestamp_min"].startswith("2019-01-01")
