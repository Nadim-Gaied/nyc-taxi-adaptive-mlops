"""Cross-month validation for canonical dataset quality reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from taxi_mlops.ingestion.inventory import month_sequence
from taxi_mlops.ingestion.yellow_taxi import DataContractError


def summarize_quality_reports(
    report_root: str | Path,
    *,
    start_month: str,
    end_month: str,
    require_complete: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Combine monthly JSON reports and evaluate hard data-contract gates."""

    report_root = Path(report_root)
    months = month_sequence(start_month, end_month)
    rows: list[dict[str, object]] = []
    missing_reports: list[str] = []

    for month in months:
        path = report_root / month[:4] / f"zone_hour_{month}.json"
        if not path.is_file():
            missing_reports.append(month)
            continue

        report = json.loads(path.read_text(encoding="utf-8"))
        source_rows = int(report["source_rows"])
        rows_in_interval = int(report["rows_in_requested_interval"])
        valid_rows = int(report["valid_rows"])
        invalid_zone_rows = int(report["invalid_zone_rows"])
        invalid_timestamp_rows = int(report["invalid_timestamp_rows"])
        expected_hours = int(report["expected_hours"])
        observed_hours = int(report["observed_source_hours"])
        missing_source_hours = list(report["missing_source_hours"])

        coverage_pass = not missing_source_hours and observed_hours >= expected_hours
        accounting_pass = valid_rows + invalid_zone_rows == rows_in_interval
        rows.append(
            {
                "month": month,
                "source_rows": source_rows,
                "rows_in_requested_interval": rows_in_interval,
                "valid_rows": valid_rows,
                "invalid_timestamp_rows": invalid_timestamp_rows,
                "invalid_zone_rows": invalid_zone_rows,
                "invalid_zone_rate": (
                    invalid_zone_rows / rows_in_interval if rows_in_interval else 0.0
                ),
                "out_of_interval_rows": int(report["out_of_interval_rows"]),
                "expected_hours": expected_hours,
                "physical_elapsed_hours": int(report["physical_elapsed_hours"]),
                "observed_source_hours": observed_hours,
                "missing_source_hour_count": len(missing_source_hours),
                "ambiguous_wall_clock_hours": int(report["ambiguous_wall_clock_hours"]),
                "nonexistent_wall_clock_hours": int(report["nonexistent_wall_clock_hours"]),
                "coverage_pass": coverage_pass,
                "accounting_pass": accounting_pass,
                "quality_pass": coverage_pass and accounting_pass,
                "report_path": str(path),
            }
        )

    if missing_reports and require_complete:
        preview = ", ".join(missing_reports[:6])
        suffix = " ..." if len(missing_reports) > 6 else ""
        raise DataContractError(
            f"{len(missing_reports)} monthly quality report(s) are missing: {preview}{suffix}"
        )

    table = pd.DataFrame(rows)
    failed_months = table.loc[~table["quality_pass"], "month"].tolist() if not table.empty else []
    summary: dict[str, object] = {
        "start_month": start_month,
        "end_month": end_month,
        "expected_report_count": len(months),
        "available_report_count": len(table),
        "missing_report_count": len(missing_reports),
        "missing_reports": missing_reports,
        "failed_quality_month_count": len(failed_months),
        "failed_quality_months": failed_months,
        "all_quality_gates_pass": not missing_reports and not failed_months,
    }
    if not table.empty:
        summary.update(
            {
                "total_source_rows": int(table["source_rows"].sum()),
                "total_valid_rows": int(table["valid_rows"].sum()),
                "total_invalid_zone_rows": int(table["invalid_zone_rows"].sum()),
                "overall_invalid_zone_rate": float(
                    table["invalid_zone_rows"].sum() / table["rows_in_requested_interval"].sum()
                ),
                "total_out_of_interval_rows": int(table["out_of_interval_rows"].sum()),
                "months_with_ambiguous_dst_hour": table.loc[
                    table["ambiguous_wall_clock_hours"] > 0, "month"
                ].tolist(),
                "months_with_nonexistent_dst_hour": table.loc[
                    table["nonexistent_wall_clock_hours"] > 0, "month"
                ].tolist(),
            }
        )
    return table, summary
