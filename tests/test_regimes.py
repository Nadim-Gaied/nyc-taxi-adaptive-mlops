from __future__ import annotations

import pandas as pd

from taxi_mlops.analysis import analyze_monthly_regimes, write_regime_figures


def _canonical_two_months() -> pd.DataFrame:
    hours = pd.date_range("2019-01-01", "2019-03-01", freq="h", inclusive="left", tz="UTC")
    rows = []
    for hour in hours:
        for zone in (1, 2):
            rows.append(
                {
                    "pickup_hour": hour,
                    "pickup_zone_id": zone,
                    "pickup_count": (hour.hour + 1) * zone,
                }
            )
    return pd.DataFrame(rows)


def test_monthly_regime_analysis_normalizes_profiles() -> None:
    monthly, profiles, similarity = analyze_monthly_regimes(
        _canonical_two_months(), timezone="UTC"
    )

    assert monthly["month"].tolist() == ["2019-01", "2019-02"]
    assert profiles.groupby("month")["normalized_demand"].mean().round(10).eq(1.0).all()
    assert similarity.loc["2019-01", "2019-02"] == 1.0


def test_regime_figures_are_written(tmp_path) -> None:
    monthly, profiles, similarity = analyze_monthly_regimes(
        _canonical_two_months(), timezone="UTC"
    )
    zone_summary = pd.DataFrame(
        {
            "volume_rank": [1, 2],
            "cumulative_demand_share": [0.7, 1.0],
        }
    )

    paths = write_regime_figures(monthly, profiles, similarity, zone_summary, tmp_path)

    assert len(paths) == 4
    assert all(path.stat().st_size > 0 for path in paths)
