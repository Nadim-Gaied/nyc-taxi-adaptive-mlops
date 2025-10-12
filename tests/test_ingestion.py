from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.ingestion.yellow_taxi import DataContractError, build_zone_hour_dataset


def test_builds_complete_zone_hour_grid_and_fills_real_zeros() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2019-01-01 00:10:00",
                "2019-01-01 00:20:00",
                "2019-01-01 01:15:00",
            ],
            "PULocationID": [1, 1, 2],
        }
    )
    zones = pd.DataFrame({"LocationID": [1, 2, 264, 265]})

    result = build_zone_hour_dataset(
        trips,
        zones,
        start="2019-01-01T00:00:00-05:00",
        end="2019-01-01T02:00:00-05:00",
        source_file="fixture.parquet",
    )

    assert len(result.data) == 4
    counts = result.data.set_index(["pickup_hour", "pickup_zone_id"])["pickup_count"]
    assert sorted(counts.tolist()) == [0, 0, 1, 2]
    assert result.data["source_hour_observed"].all()
    assert result.report.valid_rows == 3
    assert result.report.invalid_zone_rows == 0


def test_fails_instead_of_filling_an_entire_missing_source_hour() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2019-01-01 00:10:00", "2019-01-01 02:10:00"],
            "PULocationID": [1, 1],
        }
    )
    zones = pd.DataFrame({"LocationID": [1]})

    with pytest.raises(DataContractError, match="missing 1 expected source hour"):
        build_zone_hour_dataset(
            trips,
            zones,
            start="2019-01-01T00:00:00-05:00",
            end="2019-01-01T03:00:00-05:00",
            source_file="fixture.parquet",
        )


def test_requires_offset_aware_boundaries() -> None:
    trips = pd.DataFrame(
        {"tpep_pickup_datetime": ["2019-01-01 00:10:00"], "PULocationID": [1]}
    )
    zones = pd.DataFrame({"LocationID": [1]})

    with pytest.raises(DataContractError, match="explicit UTC offset"):
        build_zone_hour_dataset(
            trips,
            zones,
            start="2019-01-01 00:00:00",
            end="2019-01-01T01:00:00-05:00",
            source_file="fixture.parquet",
        )


def test_spring_dst_omits_nonexistent_wall_clock_hour() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2019-03-10 00:10:00",
                "2019-03-10 01:10:00",
                "2019-03-10 03:10:00",
            ],
            "PULocationID": [1, 1, 1],
        }
    )
    zones = pd.DataFrame({"LocationID": [1]})

    result = build_zone_hour_dataset(
        trips,
        zones,
        start="2019-03-10T00:00:00-05:00",
        end="2019-03-10T04:00:00-04:00",
        source_file="spring.parquet",
    )

    assert result.report.physical_elapsed_hours == 3
    assert result.report.expected_hours == 3
    assert result.report.nonexistent_wall_clock_hours == 1
    assert result.report.ambiguous_wall_clock_hours == 0


def test_fall_dst_omits_unidentifiable_duplicate_hour() -> None:
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2019-11-03 00:10:00",
                "2019-11-03 01:10:00",
                "2019-11-03 02:10:00",
            ],
            "PULocationID": [1, 1, 1],
        }
    )
    zones = pd.DataFrame({"LocationID": [1]})

    result = build_zone_hour_dataset(
        trips,
        zones,
        start="2019-11-03T00:00:00-04:00",
        end="2019-11-03T03:00:00-05:00",
        source_file="fall.parquet",
    )

    assert result.report.physical_elapsed_hours == 4
    assert result.report.expected_hours == 3
    assert result.report.ambiguous_wall_clock_hours == 1
    assert result.report.nonexistent_wall_clock_hours == 0
