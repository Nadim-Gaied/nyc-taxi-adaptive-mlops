"""Command-line interface for the initial vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from taxi_mlops.analysis import characterize_dataset
from taxi_mlops.baselines import daily_weekly_blend_predictions, weekly_seasonal_predictions
from taxi_mlops.datasets import load_named_split
from taxi_mlops.evaluation import regression_metrics, rolling_baseline_backtest
from taxi_mlops.ingestion import inventory_yellow_files, process_month_range
from taxi_mlops.ingestion.yellow_taxi import build_zone_hour_dataset, read_trip_file


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _build_dataset(args: argparse.Namespace) -> None:
    trips_path = Path(args.trips)
    zones_path = Path(args.zones)
    result = build_zone_hour_dataset(
        read_trip_file(trips_path),
        pd.read_csv(zones_path),
        start=args.start,
        end=args.end,
        source_file=trips_path.name,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_parquet(output, index=False)
    _write_json(Path(args.report), result.report.to_dict())
    print(f"wrote {len(result.data):,} zone-hour rows to {output}")


def _evaluate_weekly(args: argparse.Namespace) -> None:
    data = pd.read_parquet(args.dataset)
    predictions = weekly_seasonal_predictions(data, lag_hours=args.lag_hours)
    evaluation_start = pd.Timestamp(args.evaluation_start)
    if evaluation_start.tzinfo is None:
        raise ValueError("evaluation-start must include an explicit UTC offset")
    evaluation_start = evaluation_start.tz_convert("UTC")
    evaluation = predictions.loc[predictions["pickup_hour"] >= evaluation_start]
    metrics = regression_metrics(evaluation)
    metrics.update(
        {
            "model_id": f"seasonal_lag_{args.lag_hours}h",
            "evaluation_start": str(evaluation_start),
        }
    )
    _write_json(Path(args.output), metrics)
    print(json.dumps(metrics, indent=2))


def _characterize(args: argparse.Namespace) -> None:
    data = pd.read_parquet(args.dataset)
    summary, hour_of_week, zone_summary = characterize_dataset(data, timezone=args.timezone)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    hour_of_week.to_csv(output_dir / "hour_of_week.csv", index=False)
    zone_summary.to_csv(output_dir / "zone_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def _inventory_raw(args: argparse.Namespace) -> None:
    inventory = inventory_yellow_files(
        args.root,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    _write_json(Path(args.output), inventory)
    concise = {
        "expected_month_count": inventory["expected_month_count"],
        "present_month_count": inventory["present_month_count"],
        "missing_month_count": len(inventory["missing_months"]),
        "duplicate_month_count": len(inventory["duplicate_months"]),
        "output": str(args.output),
    }
    print(json.dumps(concise, indent=2))


def _evaluate_baselines(args: argparse.Namespace) -> None:
    data = pd.read_parquet(args.dataset)
    evaluation_start = pd.Timestamp(args.evaluation_start)
    if evaluation_start.tzinfo is None:
        raise ValueError("evaluation-start must include an explicit UTC offset")
    evaluation_start = evaluation_start.tz_convert("UTC")

    candidates = [
        weekly_seasonal_predictions(data, lag_hours=1),
        weekly_seasonal_predictions(data, lag_hours=24),
        weekly_seasonal_predictions(data, lag_hours=168),
        daily_weekly_blend_predictions(data, daily_weight=0.5),
    ]
    results: list[dict[str, object]] = []
    for candidate in candidates:
        evaluation = candidate.loc[candidate["pickup_hour"] >= evaluation_start]
        metrics = regression_metrics(evaluation)
        metrics["model_id"] = str(candidate["model_id"].iloc[0])
        results.append(metrics)
    results.sort(key=lambda item: float(item["mae"]))
    report = {
        "evaluation_start": str(evaluation_start),
        "evaluation_end": str(data["pickup_hour"].max()),
        "results_ranked_by_mae": results,
    }
    _write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


def _process_months(args: argparse.Namespace) -> None:
    summary = process_month_range(
        raw_root=args.raw_root,
        zones_path=args.zones,
        output_root=args.output_root,
        report_root=args.report_root,
        start_month=args.start_month,
        end_month=args.end_month,
        timezone=args.timezone,
        force=args.force,
        allow_missing=args.allow_missing,
    )
    _write_json(Path(args.summary), summary)
    concise = {
        "requested_month_count": summary["requested_month_count"],
        "processed_month_count": summary["processed_month_count"],
        "skipped_existing_month_count": summary["skipped_existing_month_count"],
        "missing_month_count": summary["missing_month_count"],
        "summary": str(args.summary),
    }
    print(json.dumps(concise, indent=2))


def _characterize_split(args: argparse.Namespace) -> None:
    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=args.split_manifest,
        split_name=args.split,
        purpose="eda",
    )
    summary, hour_of_week, zone_summary = characterize_dataset(data, timezone=args.timezone)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    hour_of_week.to_csv(output_dir / "hour_of_week.csv", index=False)
    zone_summary.to_csv(output_dir / "zone_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def _backtest_baselines_split(args: argparse.Namespace) -> None:
    purpose = "development" if args.split == "development" else "validation"
    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=args.split_manifest,
        split_name=args.split,
        purpose=purpose,
    )
    weekly, summary = rolling_baseline_backtest(
        data,
        evaluation_start=args.evaluation_start,
        timezone=args.timezone,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_dir / "weekly_metrics.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taxi-mlops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="build the canonical zone-hour table")
    build.add_argument("--trips", required=True)
    build.add_argument("--zones", required=True)
    build.add_argument("--start", required=True)
    build.add_argument("--end", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--report", required=True)
    build.set_defaults(handler=_build_dataset)

    evaluate = subparsers.add_parser(
        "evaluate-weekly-baseline", help="evaluate a weekly seasonal baseline"
    )
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--evaluation-start", required=True)
    evaluate.add_argument("--lag-hours", type=int, default=168)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_evaluate_weekly)

    characterize = subparsers.add_parser(
        "characterize", help="produce decision-oriented temporal and spatial summaries"
    )
    characterize.add_argument("--dataset", required=True)
    characterize.add_argument("--output-dir", required=True)
    characterize.add_argument("--timezone", default="America/New_York")
    characterize.set_defaults(handler=_characterize)

    inventory = subparsers.add_parser(
        "inventory-raw", help="inventory local monthly TLC Parquet files"
    )
    inventory.add_argument("--root", required=True)
    inventory.add_argument("--start-month", required=True)
    inventory.add_argument("--end-month", required=True)
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(handler=_inventory_raw)

    baselines = subparsers.add_parser(
        "evaluate-baselines", help="compare leakage-safe persistence and seasonal baselines"
    )
    baselines.add_argument("--dataset", required=True)
    baselines.add_argument("--evaluation-start", required=True)
    baselines.add_argument("--output", required=True)
    baselines.set_defaults(handler=_evaluate_baselines)

    process = subparsers.add_parser(
        "process-months", help="resumably build canonical datasets for a month range"
    )
    process.add_argument("--raw-root", required=True)
    process.add_argument("--zones", required=True)
    process.add_argument("--output-root", required=True)
    process.add_argument("--report-root", required=True)
    process.add_argument("--start-month", required=True)
    process.add_argument("--end-month", required=True)
    process.add_argument("--summary", required=True)
    process.add_argument("--timezone", default="America/New_York")
    process.add_argument("--force", action="store_true")
    process.add_argument("--allow-missing", action="store_true")
    process.set_defaults(handler=_process_months)

    split_characterize = subparsers.add_parser(
        "characterize-split", help="characterize a complete, access-controlled split"
    )
    split_characterize.add_argument("--monthly-root", required=True)
    split_characterize.add_argument("--split-manifest", required=True)
    split_characterize.add_argument("--split", default="development")
    split_characterize.add_argument("--output-dir", required=True)
    split_characterize.add_argument("--timezone", default="America/New_York")
    split_characterize.set_defaults(handler=_characterize_split)

    backtest = subparsers.add_parser(
        "backtest-baselines-split",
        help="run rolling weekly baseline evaluation on a complete declared split",
    )
    backtest.add_argument("--monthly-root", required=True)
    backtest.add_argument("--split-manifest", required=True)
    backtest.add_argument("--split", default="development")
    backtest.add_argument("--output-dir", required=True)
    backtest.add_argument("--evaluation-start")
    backtest.add_argument("--timezone", default="America/New_York")
    backtest.set_defaults(handler=_backtest_baselines_split)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
