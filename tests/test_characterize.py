from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.analysis import characterize_dataset


def test_characterization_measures_recurrence_and_spatial_concentration() -> None:
    hours = pd.date_range("2019-01-01", periods=24 * 15, freq="h", tz="UTC")
    rows: list[dict[str, object]] = []
    for pickup_hour in hours:
        hour_signal = pickup_hour.hour + 1
        rows.extend(
            [
                {
                    "pickup_hour": pickup_hour,
                    "pickup_zone_id": 1,
                    "pickup_count": hour_signal * 10,
                },
                {
                    "pickup_hour": pickup_hour,
                    "pickup_zone_id": 2,
                    "pickup_count": hour_signal,
                },
            ]
        )

    summary, hour_of_week, zone_summary = characterize_dataset(pd.DataFrame(rows), timezone="UTC")

    assert summary["dataset"]["zones"] == 2
    assert summary["autocorrelation"]["lag_24h"] == pytest.approx(1.0)
    assert summary["autocorrelation"]["lag_168h"] == pytest.approx(1.0)
    assert len(hour_of_week) == 168
    assert zone_summary.iloc[0]["pickup_zone_id"] == 1
    assert summary["spatial_concentration"]["top_10_zone_share"] == pytest.approx(1.0)


def test_characterization_rejects_incomplete_grid() -> None:
    data = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(
                ["2019-01-01T00:00:00Z", "2019-01-01T01:00:00Z", "2019-01-01T01:00:00Z"]
            ),
            "pickup_zone_id": [1, 1, 2],
            "pickup_count": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="incomplete"):
        characterize_dataset(data)

