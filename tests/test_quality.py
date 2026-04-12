from __future__ import annotations

import json

import pytest

from taxi_mlops.ingestion import summarize_quality_reports
from taxi_mlops.ingestion.yellow_taxi import DataContractError


def _write_report(root, month: str, *, missing_hours: list[str] | None = None) -> None:
    report = {
        "source_rows": 100,
        "rows_in_requested_interval": 98,
        "valid_rows": 95,
        "invalid_timestamp_rows": 0,
        "invalid_zone_rows": 3,
        "out_of_interval_rows": 2,
        "expected_hours": 744,
        "physical_elapsed_hours": 744,
        "observed_source_hours": 744,
        "missing_source_hours": missing_hours or [],
        "ambiguous_wall_clock_hours": 0,
        "nonexistent_wall_clock_hours": 0,
    }
    path = root / month[:4] / f"zone_hour_{month}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_quality_summary_combines_reports_and_metrics(tmp_path) -> None:
    _write_report(tmp_path, "2019-01")
    _write_report(tmp_path, "2019-02")

    table, summary = summarize_quality_reports(tmp_path, start_month="2019-01", end_month="2019-02")

    assert table["quality_pass"].all()
    assert summary["all_quality_gates_pass"] is True
    assert summary["total_source_rows"] == 200
    assert summary["overall_invalid_zone_rate"] == pytest.approx(3 / 98)


def test_quality_summary_refuses_missing_reports_when_complete_required(tmp_path) -> None:
    _write_report(tmp_path, "2019-01")

    with pytest.raises(DataContractError, match="1 monthly quality report"):
        summarize_quality_reports(tmp_path, start_month="2019-01", end_month="2019-02")


def test_quality_summary_flags_missing_source_hours(tmp_path) -> None:
    _write_report(tmp_path, "2019-01", missing_hours=["2019-01-02 05:00:00+00:00"])

    table, summary = summarize_quality_reports(tmp_path, start_month="2019-01", end_month="2019-01")

    assert table.loc[0, "quality_pass"] == False  # noqa: E712
    assert summary["failed_quality_months"] == ["2019-01"]
