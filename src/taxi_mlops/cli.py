"""Command-line interface for the initial vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from taxi_mlops.analysis import characterize_dataset
from taxi_mlops.baselines import daily_weekly_blend_predictions, weekly_seasonal_predictions
from taxi_mlops.evaluation import regression_metrics
from taxi_mlops.ingestion import inventory_yellow_files
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
