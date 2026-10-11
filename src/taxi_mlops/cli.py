"""Command-line interface for the initial vertical slice."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from taxi_mlops.analysis import (
    analyze_monthly_regimes,
    characterize_dataset,
    write_regime_figures,
)
from taxi_mlops.baselines import daily_weekly_blend_predictions, weekly_seasonal_predictions
from taxi_mlops.datasets import load_named_split, split_months
from taxi_mlops.evaluation import (
    regression_metrics,
    rolling_baseline_backtest,
    write_baseline_backtest_figures,
    write_window_experiment_figures,
)
from taxi_mlops.features import build_feature_table
from taxi_mlops.ingestion import (
    inventory_yellow_files,
    process_month_range,
    summarize_quality_reports,
)
from taxi_mlops.ingestion.yellow_taxi import build_zone_hour_dataset, read_trip_file
from taxi_mlops.models import (
    run_feature_ablation_experiment,
    run_model_family_experiment,
    run_seasonal_gap_experiment,
    run_training_history_experiment,
    run_training_window_experiment,
)
from taxi_mlops.policies import (
    build_alert_scorecard,
    build_promotion_scorecard,
    load_lifecycle_policy,
    load_locked_replay_plan,
    load_policy_validation_plan,
    run_complete_baseline_validation,
    summarize_policy_validation,
)
from taxi_mlops.production import (
    ForecastAlertManager,
    HourlyPredictionPipeline,
    HourlyScoringPipeline,
    MLflowLightGBMResolver,
    ProductionBackfillRunner,
    ProductionCycleRunner,
    ProductionIntegrityAuditor,
    ProductionMonitoringReporter,
    ProductionStateStore,
    ScheduledTrainingPipeline,
    bootstrap_production_reference,
    create_control_plane_checkpoint,
    load_production_config,
    verify_control_plane_checkpoint,
)
from taxi_mlops.replay import (
    ModelRegistry,
    MonthlyReplayDataSource,
    ReplayModelStore,
    build_monthly_feature_store,
    run_locked_replay,
    seed_replay_registry,
    write_locked_replay_report,
)
from taxi_mlops.tracking import import_locked_replay_to_mlflow


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


def _summarize_quality(args: argparse.Namespace) -> None:
    table, summary = summarize_quality_reports(
        args.report_root,
        start_month=args.start_month,
        end_month=args.end_month,
        require_complete=not args.allow_missing,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "monthly_quality.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
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
    write_baseline_backtest_figures(weekly, output_dir / "figures")
    print(json.dumps(summary, indent=2))


def _window_experiment_split(args: argparse.Namespace) -> None:
    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=args.split_manifest,
        split_name="development",
        purpose="development",
    )
    features = build_feature_table(data, timezone=args.timezone)
    results, predictions = run_training_window_experiment(
        features,
        cutoffs=args.cutoffs,
        windows_days=tuple(args.windows_days),
        include_expanding=not args.no_expanding,
        evaluation_days=args.evaluation_days,
        timezone=args.timezone,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "window_results.csv", index=False)
    predictions.to_parquet(output_dir / "window_predictions.parquet", index=False)
    figures = write_window_experiment_figures(results, output_dir / "figures")
    concise = {
        "cutoffs": args.cutoffs,
        "windows_days": args.windows_days,
        "include_expanding": not args.no_expanding,
        "evaluation_days": args.evaluation_days,
        "experiment_count": len(results),
        "figures": [str(path) for path in figures],
        "best_mean_mae_window": str(results.groupby("training_window")["mae"].mean().idxmin()),
    }
    _write_json(output_dir / "summary.json", concise)
    print(json.dumps(concise, indent=2))


def _feature_ablation_split(args: argparse.Namespace) -> None:
    with Path(args.experiment_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation_config = config["evaluation"]
    tier_config = config["demand_tiers"]
    feature_groups = config["feature_groups"]
    feature_sets = {
        name: [column for group_name in group_names for column in feature_groups[group_name]]
        for name, group_names in config["feature_sets"].items()
    }
    if args.feature_sets:
        unknown = sorted(set(args.feature_sets).difference(feature_sets))
        if unknown:
            raise ValueError(f"unknown feature sets: {', '.join(unknown)}")
        feature_sets = {name: feature_sets[name] for name in args.feature_sets}

    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=config["split_manifest"],
        split_name="development",
        purpose="development",
    )
    features = build_feature_table(data, timezone=config["timezone"])
    results, tier_results = run_feature_ablation_experiment(
        features,
        cutoffs=evaluation_config["origins"],
        feature_sets=feature_sets,
        training_window_days=args.training_window_days,
        evaluation_days=evaluation_config["horizon_calendar_days"],
        timezone=config["timezone"],
        tier_boundaries=tier_config["boundaries"],
        tier_labels=tier_config["labels"],
        checkpoint_dir=Path(args.output_dir) / "checkpoints",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "feature_ablation_results.csv", index=False)
    tier_results.to_csv(output_dir / "feature_ablation_tier_results.csv", index=False)
    aggregate = (
        results.groupby("feature_set")
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_mae=("mae", "mean"),
            mean_zone_mae_p95=("zone_mae_p95", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
        )
        .sort_values("mean_wape")
        .reset_index()
    )
    aggregate.to_csv(output_dir / "feature_ablation_summary.csv", index=False)
    tier_summary = (
        tier_results.groupby(["feature_set", "demand_tier"])
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_zones=("zones", "mean"),
        )
        .reset_index()
    )
    tier_summary.to_csv(output_dir / "feature_ablation_tier_summary.csv", index=False)

    reference_feature_set = config["reference"]["feature_set"]
    paired_summary: list[dict[str, object]] = []
    if reference_feature_set in set(results["feature_set"]):
        reference = results.loc[results["feature_set"].eq(reference_feature_set)].set_index(
            "cutoff"
        )
        for feature_set_name, candidate in results.groupby("feature_set"):
            candidate = candidate.set_index("cutoff").loc[reference.index]
            wape_delta = candidate["wape"] - reference["wape"]
            paired_summary.append(
                {
                    "feature_set": feature_set_name,
                    "reference_feature_set": reference_feature_set,
                    "mean_wape": float(candidate["wape"].mean()),
                    "relative_wape_change_percent": float(
                        100 * wape_delta.mean() / reference["wape"].mean()
                    ),
                    "origins_beating_reference": int(wape_delta.lt(0).sum()),
                    "origins_tied_reference": int(wape_delta.abs().lt(1e-12).sum()),
                    "worst_origin_wape_delta_percentage_points": float(100 * wape_delta.max()),
                }
            )
        pd.DataFrame(paired_summary).sort_values("mean_wape").to_csv(
            output_dir / "feature_ablation_paired_summary.csv", index=False
        )
    report = {
        "experiment_config": str(args.experiment_config),
        "training_window_days": args.training_window_days,
        "origin_count": len(evaluation_config["origins"]),
        "feature_set_count": len(feature_sets),
        "best_mean_wape_feature_set": str(aggregate.iloc[0]["feature_set"]),
        "reference_feature_set": reference_feature_set,
    }
    _write_json(output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


def _model_family_experiment_split(args: argparse.Namespace) -> None:
    with Path(args.experiment_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation_config = config["evaluation"]
    tier_config = config["demand_tiers"]
    feature_groups = config["feature_groups"]
    selected_feature_set = config["reference"]["feature_set"]
    selected_groups = config["feature_sets"][selected_feature_set]
    feature_columns = [
        column for group_name in selected_groups for column in feature_groups[group_name]
    ]
    model_names = args.models or config["candidate_models"]

    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=config["split_manifest"],
        split_name="development",
        purpose="development",
    )
    features = build_feature_table(data, timezone=config["timezone"])
    results, tier_results = run_model_family_experiment(
        features,
        cutoffs=evaluation_config["origins"],
        feature_columns=feature_columns,
        model_names=model_names,
        training_window_days=args.training_window_days,
        evaluation_days=evaluation_config["horizon_calendar_days"],
        timezone=config["timezone"],
        tier_boundaries=tier_config["boundaries"],
        tier_labels=tier_config["labels"],
        model_parameters=config.get("model_parameters"),
        checkpoint_dir=Path(args.output_dir) / "checkpoints",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "model_family_results.csv", index=False)
    tier_results.to_csv(output_dir / "model_family_tier_results.csv", index=False)
    aggregate = (
        results.groupby("model_name")
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_mae=("mae", "mean"),
            mean_zone_mae_p95=("zone_mae_p95", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
        )
        .sort_values("mean_wape")
        .reset_index()
    )
    aggregate.to_csv(output_dir / "model_family_summary.csv", index=False)
    tier_summary = (
        tier_results.groupby(["model_name", "demand_tier"])
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_zones=("zones", "mean"),
        )
        .reset_index()
    )
    tier_summary.to_csv(output_dir / "model_family_tier_summary.csv", index=False)
    report = {
        "experiment_config": str(args.experiment_config),
        "feature_set": selected_feature_set,
        "training_window_days": args.training_window_days,
        "origin_count": len(evaluation_config["origins"]),
        "model_names": list(model_names),
        "best_mean_wape_model": str(aggregate.iloc[0]["model_name"]),
    }
    _write_json(output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


def _poisson_screen_split(args: argparse.Namespace) -> None:
    with Path(args.experiment_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    screen_config = config["poisson_screen"]
    evaluation_config = config["evaluation"]
    tier_config = config["demand_tiers"]
    feature_groups = config["feature_groups"]
    selected_feature_set = config["reference"]["feature_set"]
    selected_groups = config["feature_sets"][selected_feature_set]
    feature_columns = [
        column for group_name in selected_groups for column in feature_groups[group_name]
    ]

    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=config["split_manifest"],
        split_name="development",
        purpose="development",
    )
    features = build_feature_table(data, timezone=config["timezone"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frames: list[pd.DataFrame] = []
    tier_frames: list[pd.DataFrame] = []
    for alpha in screen_config["alphas"]:
        parameter_id = f"alpha_{float(alpha):g}".replace(".", "p")
        parameters = {
            "poisson_regression": {
                "alpha": float(alpha),
                "solver": screen_config["solver"],
                "max_iter": int(screen_config["max_iter"]),
                "tol": float(screen_config["tol"]),
            }
        }
        results, tier_results = run_model_family_experiment(
            features,
            cutoffs=screen_config["origins"],
            feature_columns=feature_columns,
            model_names=["poisson_regression"],
            training_window_days=args.training_window_days,
            evaluation_days=evaluation_config["horizon_calendar_days"],
            timezone=config["timezone"],
            tier_boundaries=tier_config["boundaries"],
            tier_labels=tier_config["labels"],
            model_parameters=parameters,
            checkpoint_dir=output_dir / "checkpoints" / parameter_id,
        )
        results["parameter_id"] = parameter_id
        results["alpha"] = float(alpha)
        tier_results["parameter_id"] = parameter_id
        tier_results["alpha"] = float(alpha)
        result_frames.append(results)
        tier_frames.append(tier_results)

    all_results = pd.concat(result_frames, ignore_index=True)
    all_tiers = pd.concat(tier_frames, ignore_index=True)
    all_results.to_csv(output_dir / "poisson_screen_results.csv", index=False)
    all_tiers.to_csv(output_dir / "poisson_screen_tier_results.csv", index=False)
    summary = (
        all_results.groupby(["parameter_id", "alpha"])
        .agg(
            origins=("cutoff", "size"),
            converged_origins=("hit_iteration_limit", lambda values: int((~values).sum())),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_mae=("mae", "mean"),
            mean_bias=("mean_bias", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
            mean_iterations=("iterations", "mean"),
        )
        .sort_values("mean_wape")
        .reset_index()
    )
    summary.to_csv(output_dir / "poisson_screen_summary.csv", index=False)
    eligible = summary.loc[summary["converged_origins"].eq(len(screen_config["origins"]))]
    report = {
        "experiment_config": str(args.experiment_config),
        "feature_set": selected_feature_set,
        "training_window_days": args.training_window_days,
        "origins": screen_config["origins"],
        "alphas": screen_config["alphas"],
        "best_mean_wape_parameter": str(summary.iloc[0]["parameter_id"]),
        "best_fully_converged_parameter": (
            str(eligible.iloc[0]["parameter_id"]) if not eligible.empty else None
        ),
    }
    _write_json(output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


def _training_history_experiment_split(args: argparse.Namespace) -> None:
    with Path(args.experiment_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation_config = config["evaluation"]
    tier_config = config["demand_tiers"]
    feature_groups = config["feature_groups"]
    selected_feature_set = config["reference"]["feature_set"]
    selected_groups = config["feature_sets"][selected_feature_set]
    feature_columns = [
        column for group_name in selected_groups for column in feature_groups[group_name]
    ]
    strategies = args.strategies or config["candidate_training_histories"]

    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=config["split_manifest"],
        split_name="development",
        purpose="development",
    )
    features = build_feature_table(data, timezone=config["timezone"])
    results, tier_results, weekly_results = run_training_history_experiment(
        features,
        cutoffs=evaluation_config["origins"],
        feature_columns=feature_columns,
        strategies=strategies,
        evaluation_days=evaluation_config["horizon_calendar_days"],
        timezone=config["timezone"],
        tier_boundaries=tier_config["boundaries"],
        tier_labels=tier_config["labels"],
        model_parameters=config.get("model_parameters", {}).get("lightgbm_poisson"),
        checkpoint_dir=Path(args.output_dir) / "checkpoints",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "training_history_results.csv", index=False)
    tier_results.to_csv(output_dir / "training_history_tier_results.csv", index=False)
    weekly_results.to_csv(output_dir / "training_history_weekly_results.csv", index=False)

    aggregate = (
        results.groupby("training_history")
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_origin_wape=("wape", "max"),
            mean_mae=("mae", "mean"),
            mean_zone_mae_p95=("zone_mae_p95", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
        )
        .reset_index()
    )
    weekly_summary = (
        weekly_results.groupby("training_history")
        .agg(
            mean_weekly_wape=("wape", "mean"),
            p90_weekly_wape=("wape", lambda values: float(values.quantile(0.90))),
            worst_weekly_wape=("wape", "max"),
        )
        .reset_index()
    )
    aggregate = aggregate.merge(
        weekly_summary, on="training_history", how="left", validate="1:1"
    ).sort_values("mean_wape")
    aggregate.to_csv(output_dir / "training_history_summary.csv", index=False)

    tier_summary = (
        tier_results.groupby(["training_history", "demand_tier"])
        .agg(
            origins=("cutoff", "size"),
            mean_wape=("wape", "mean"),
            worst_wape=("wape", "max"),
            mean_zones=("zones", "mean"),
        )
        .reset_index()
    )
    tier_summary.to_csv(output_dir / "training_history_tier_summary.csv", index=False)

    reference_name = "recent_365d"
    paired_rows: list[dict[str, object]] = []
    if reference_name in set(results["training_history"]):
        reference = results.loc[results["training_history"].eq(reference_name)].set_index("cutoff")
        for strategy, candidate in results.groupby("training_history"):
            candidate = candidate.set_index("cutoff").loc[reference.index]
            delta = candidate["wape"] - reference["wape"]
            paired_rows.append(
                {
                    "training_history": strategy,
                    "reference": reference_name,
                    "relative_wape_change_percent": float(
                        100 * delta.mean() / reference["wape"].mean()
                    ),
                    "origins_beating_reference": int(delta.lt(0).sum()),
                    "origins_tied_reference": int(delta.abs().lt(1e-12).sum()),
                    "worst_origin_delta_percentage_points": float(100 * delta.max()),
                }
            )
        pd.DataFrame(paired_rows).sort_values("relative_wape_change_percent").to_csv(
            output_dir / "training_history_paired_summary.csv", index=False
        )
    report = {
        "experiment_config": str(args.experiment_config),
        "feature_set": selected_feature_set,
        "origin_count": len(evaluation_config["origins"]),
        "strategies": list(strategies),
        "best_mean_wape_history": str(aggregate.iloc[0]["training_history"]),
        "reference_history": reference_name,
    }
    _write_json(output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


def _policy_validation_split(args: argparse.Namespace) -> None:
    plan_path = Path(args.validation_config)
    plan = load_policy_validation_plan(plan_path)
    project_root = plan_path.resolve().parent.parent

    def resolve_plan_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else project_root / path

    policy = load_lifecycle_policy(resolve_plan_path(plan["lifecycle_policy"]))
    with resolve_plan_path(plan["development_experiment"]).open(encoding="utf-8") as handle:
        development_config = yaml.safe_load(handle)

    evaluation_config = plan["evaluation"]
    model_config = plan["model"]
    access_config = plan["data_access"]
    tier_config = development_config["demand_tiers"]
    feature_groups = development_config["feature_groups"]
    selected_feature_set = model_config["feature_set"]
    if selected_feature_set != policy["production_candidate"]["feature_set"]:
        raise ValueError("validation feature set differs from the frozen lifecycle policy")
    selected_groups = development_config["feature_sets"][selected_feature_set]
    feature_columns = [
        column for group_name in selected_groups for column in feature_groups[group_name]
    ]
    reference_history = str(model_config["reference_history"])
    candidate_histories = [str(value) for value in model_config["candidate_histories"]]
    strategies = [reference_history, *candidate_histories]

    split_manifest = resolve_plan_path(plan["split_manifest"])
    data_frames = [
        load_named_split(
            monthly_root=args.monthly_root,
            manifest_path=split_manifest,
            split_name=split_name,
            purpose=str(access_config["purpose"]),
        )
        for split_name in access_config["history_splits"]
    ]
    data = (
        pd.concat(data_frames, ignore_index=True)
        .sort_values(["pickup_hour", "pickup_zone_id"])
        .reset_index(drop=True)
    )
    features = build_feature_table(data, timezone=str(plan["timezone"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, tier_results, weekly_results = run_training_history_experiment(
        features,
        cutoffs=evaluation_config["origins"],
        feature_columns=feature_columns,
        strategies=strategies,
        evaluation_days=int(evaluation_config["horizon_calendar_days"]),
        timezone=str(plan["timezone"]),
        tier_boundaries=tier_config["boundaries"],
        tier_labels=tier_config["labels"],
        model_parameters=development_config.get("model_parameters", {}).get("lightgbm_poisson"),
        checkpoint_dir=output_dir / "checkpoints",
    )
    baseline_results, baseline_tiers, baseline_weeks = run_complete_baseline_validation(
        data,
        cutoffs=evaluation_config["origins"],
        evaluation_days=int(evaluation_config["horizon_calendar_days"]),
        timezone=str(plan["timezone"]),
        tier_boundaries=tier_config["boundaries"],
        tier_labels=tier_config["labels"],
    )
    results = pd.concat([results, baseline_results], ignore_index=True).sort_values(
        ["cutoff", "wape", "training_history"]
    )
    tier_results = pd.concat([tier_results, baseline_tiers], ignore_index=True).sort_values(
        ["cutoff", "training_history", "demand_tier"]
    )
    weekly_results = pd.concat([weekly_results, baseline_weeks], ignore_index=True).sort_values(
        ["cutoff", "training_history", "week_index"]
    )

    alert_scorecard = build_alert_scorecard(
        weekly_results, policy, reference_history=reference_history
    )
    promotion_scorecard = build_promotion_scorecard(
        results,
        tier_results,
        alert_scorecard,
        policy,
        reference_history=reference_history,
        candidate_histories=candidate_histories,
    )
    summary = summarize_policy_validation(
        results,
        alert_scorecard,
        promotion_scorecard,
        reference_history=reference_history,
    )
    summary["validation_config"] = str(plan_path)
    summary["locked_replay_opened"] = False

    results.to_csv(output_dir / "policy_validation_results.csv", index=False)
    tier_results.to_csv(output_dir / "policy_validation_tier_results.csv", index=False)
    weekly_results.to_csv(output_dir / "policy_validation_weekly_results.csv", index=False)
    alert_scorecard.to_csv(output_dir / "policy_validation_alerts.csv", index=False)
    promotion_scorecard.to_csv(output_dir / "policy_validation_promotions.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _build_replay_feature_store(args: argparse.Namespace) -> None:
    report = build_monthly_feature_store(
        canonical_root=args.canonical_root,
        feature_root=args.feature_root,
        start_month=args.start_month,
        end_month=args.end_month,
        timezone=args.timezone,
        force=args.force,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    summary = {
        "start_month": args.start_month,
        "end_month": args.end_month,
        "months": int(len(report)),
        "written": int(report["status"].eq("written").sum()),
        "skipped_existing": int(report["status"].eq("skipped_existing").sum()),
        "report": str(output),
    }
    print(json.dumps(summary, indent=2))


def _seed_replay_registry(args: argparse.Namespace) -> None:
    plan_path = Path(args.replay_config)
    plan = load_locked_replay_plan(plan_path)
    project_root = plan_path.resolve().parent.parent

    def resolve_plan_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else project_root / path

    with resolve_plan_path("configs/development_experiments.yaml").open(encoding="utf-8") as handle:
        development_config = yaml.safe_load(handle)
    feature_set = development_config["reference"]["feature_set"]
    feature_groups = development_config["feature_groups"]
    selected_groups = development_config["feature_sets"][feature_set]
    feature_columns = [
        column for group_name in selected_groups for column in feature_groups[group_name]
    ]
    split_manifest = resolve_plan_path(plan["split_manifest"])
    _, history_start, _ = split_months(split_manifest, "development")
    _, replay_start, _ = split_months(split_manifest, "locked_replay")
    source = MonthlyReplayDataSource(
        feature_root=args.feature_root,
        start=history_start,
        end=replay_start,
        timezone=plan["timezone"],
    )
    manifest_path = Path(args.output_dir) / "registry.json"
    registry = (
        ModelRegistry.from_manifest(manifest_path) if manifest_path.is_file() else ModelRegistry()
    )
    store = ReplayModelStore(Path(args.output_dir) / "models", registry)
    initialization = plan["initialization"]
    report = seed_replay_registry(
        source=source,
        model_store=store,
        origins=initialization["seed_origins"],
        histories=initialization["seed_histories"],
        feature_columns=feature_columns,
        feature_set=feature_set,
        timezone=plan["timezone"],
        model_parameters=development_config.get("model_parameters", {}).get("lightgbm_poisson"),
        manifest_path=manifest_path,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "seed_report.csv", index=False)
    summary = {
        "models": len(registry.records()),
        "trained": int(report["status"].eq("trained").sum()),
        "resumed": int(report["status"].eq("resumed").sum()),
        "registry": str(manifest_path),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _run_locked_replay(args: argparse.Namespace) -> None:
    plan_path = Path(args.replay_config)
    plan = load_locked_replay_plan(plan_path)
    project_root = plan_path.resolve().parent.parent

    def resolve_plan_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else project_root / path

    policy = load_lifecycle_policy(resolve_plan_path(plan["lifecycle_policy"]))
    with resolve_plan_path("configs/development_experiments.yaml").open(encoding="utf-8") as handle:
        development_config = yaml.safe_load(handle)
    feature_set = development_config["reference"]["feature_set"]
    feature_groups = development_config["feature_groups"]
    selected_groups = development_config["feature_sets"][feature_set]
    feature_columns = [
        column for group_name in selected_groups for column in feature_groups[group_name]
    ]
    _, history_start, _ = split_months(resolve_plan_path(plan["split_manifest"]), "development")
    comparison = run_locked_replay(
        feature_root=args.feature_root,
        output_dir=args.output_dir,
        seed_registry_path=args.seed_registry,
        plan=plan,
        policy=policy,
        history_start=history_start,
        feature_columns=feature_columns,
        feature_set=feature_set,
        model_parameters=development_config.get("model_parameters", {}).get("lightgbm_poisson"),
        resume_completed=args.resume_completed,
    )
    print(comparison.to_string(index=False))


def _summarize_locked_replay(args: argparse.Namespace) -> None:
    summary = write_locked_replay_report(args.replay_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


def _import_locked_replay_mlflow(args: argparse.Namespace) -> None:
    result = import_locked_replay_to_mlflow(
        replay_dir=args.replay_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        archive_ref=args.archive_ref,
        project_root=args.project_root,
        registered_model_name=args.registered_model_name,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.to_dict(), indent=2))


def _predict_production_hour(args: argparse.Namespace) -> None:
    state = ProductionStateStore(args.state_db)
    resolver = MLflowLightGBMResolver(
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        alias=args.model_alias,
    )
    pipeline = HourlyPredictionPipeline(
        feature_root=args.feature_root,
        output_root=args.output_root,
        state_store=state,
        model_resolver=resolver,
        timezone=args.timezone,
        expected_zones=args.expected_zones,
    )
    print(json.dumps(asdict(pipeline.run(args.target_hour)), indent=2))


def _score_production_hour(args: argparse.Namespace) -> None:
    state = ProductionStateStore(args.state_db)
    pipeline = HourlyScoringPipeline(
        feature_root=args.feature_root,
        output_root=args.output_root,
        state_store=state,
        timezone=args.timezone,
    )
    print(json.dumps(asdict(pipeline.run(args.target_hour, as_of=args.as_of)), indent=2))


def _run_production_hour(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    resolver = MLflowLightGBMResolver(
        tracking_uri=config.tracking.uri,
        model_name=config.tracking.registered_model_name,
        alias=config.tracking.production_alias,
    )
    prediction_pipeline = HourlyPredictionPipeline(
        feature_root=config.storage.feature_root,
        output_root=config.storage.prediction_root,
        state_store=state,
        model_resolver=resolver,
        timezone=config.timezone,
        expected_zones=config.expected_zones,
    )
    result: dict[str, object] = {"prediction": asdict(prediction_pipeline.run(args.target_hour))}
    if args.score_as_of:
        scoring_pipeline = HourlyScoringPipeline(
            feature_root=config.storage.feature_root,
            output_root=config.storage.score_root,
            state_store=state,
            timezone=config.timezone,
        )
        result["score"] = asdict(scoring_pipeline.run(args.target_hour, as_of=args.score_as_of))
    print(json.dumps(result, indent=2))


def _train_production_model(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    pipeline = ScheduledTrainingPipeline(config=config, state_store=state)
    print(json.dumps(pipeline.run(args.as_of, force=args.force).to_dict(), indent=2))


def _build_production_monitoring(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    reporter = ProductionMonitoringReporter(config=config, state_store=state)
    print(json.dumps(reporter.build(as_of=args.as_of).to_dict(), indent=2))


def _serve_dashboard(args: argparse.Namespace) -> None:
    from taxi_mlops.dashboard import serve_dashboard

    root = Path(args.project_root).resolve()
    config = load_production_config(root / args.config, project_root=root)
    replay_config = None
    if args.replay_config:
        candidate = load_production_config(root / args.replay_config, project_root=root)
        if candidate.storage.state_db.is_file():
            replay_config = candidate
    serve_dashboard(
        config,
        html_path=root / "docs" / "dashboard.html",
        replay_config=replay_config,
        host=args.host,
        port=args.port,
    )


def _build_dashboard_index(args: argparse.Namespace) -> None:
    from taxi_mlops.production.dashboard_index import build_zone_history_index

    root = Path(args.project_root).resolve()
    config = load_production_config(root / args.config, project_root=root)
    print(json.dumps(build_zone_history_index(config), indent=2))


def _run_continuous_replay(args: argparse.Namespace) -> None:
    from taxi_mlops.production.continuous_replay import ContinuousReplay

    root = Path(args.project_root).resolve()
    config = load_production_config(root / args.config, project_root=root)
    replay = ContinuousReplay(config, start=args.start, end=args.end)
    for step in replay.run(interval_seconds=args.interval_seconds, max_ticks=args.max_ticks):
        print(json.dumps(asdict(step)), flush=True)


def _list_production_alerts(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    incidents = state.alert_history(active_only=args.active_only)
    payload = []
    for incident in incidents:
        item = asdict(incident)
        if args.include_history:
            item["transitions"] = [
                asdict(value) for value in state.alert_transition_history(incident.incident_id)
            ]
            item["deliveries"] = state.alert_delivery_history(incident.incident_id)
        payload.append(item)
    print(json.dumps({"incidents": payload}, indent=2))


def _acknowledge_production_alert(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    manager = ForecastAlertManager.from_config(config=config, state_store=state)
    acknowledged_at = args.at or pd.Timestamp.now(tz="UTC").isoformat()
    result = manager.acknowledge(
        args.incident_id,
        actor=args.actor,
        note=args.note,
        acknowledged_at=acknowledged_at,
    )
    print(json.dumps(result.to_dict(), indent=2))


def _audit_production_state(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    auditor = ProductionIntegrityAuditor(config=config, state_store=state)
    report = auditor.run(verify_registry=not args.skip_registry)
    print(json.dumps(report.to_dict(), indent=2))
    if report.status == "failed":
        raise SystemExit(1)


def _checkpoint_production_state(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    result = create_control_plane_checkpoint(
        config=config,
        state_store=state,
        destination=args.destination,
    )
    print(json.dumps(result.to_dict(), indent=2))


def _run_production_cycle(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    cutoff = args.as_of or pd.Timestamp.now(tz="UTC").floor("h").isoformat()
    runner = ProductionCycleRunner(config=config, state_store=state)
    result = runner.run(
        cutoff,
        allow_missing_previous_prediction=args.allow_missing_previous_prediction,
    )
    print(json.dumps(result.to_dict(), indent=2))


def _backfill_production(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    runner = ProductionBackfillRunner(config=config, state_store=state)
    result = runner.run(
        start=args.start,
        end=args.end,
        maximum_hours=args.maximum_hours,
    )
    print(json.dumps(result.to_dict(), indent=2))


def _extend_dashboard_history(args: argparse.Namespace) -> None:
    from taxi_mlops.production.history_extension import (
        extend_historical_scores,
        plan_historical_epochs,
    )

    root = Path(args.project_root).resolve()
    registry = root / args.registry
    epochs = plan_historical_epochs(registry, start=args.start, end=args.end)
    if args.plan_only:
        print(
            json.dumps(
                [
                    {
                        "start": epoch.start.isoformat(),
                        "end": epoch.end.isoformat(),
                        "model_id": epoch.model_id,
                        "model_path": str(epoch.model_path),
                        "model_exists": epoch.model_path.is_file(),
                    }
                    for epoch in epochs
                ],
                indent=2,
            )
        )
        return
    config = load_production_config(root / args.config, project_root=root)
    state = ProductionStateStore(config.storage.state_db)
    results = extend_historical_scores(
        config=config,
        state=state,
        registry_path=registry,
        start=args.start,
        end=args.end,
        maximum_hours=args.maximum_hours,
    )
    print(json.dumps(results, indent=2))


def _bootstrap_production_reference(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    state = ProductionStateStore(config.storage.state_db)
    result = bootstrap_production_reference(
        config=config,
        state_store=state,
        model_path=args.model_path,
        training_start=args.training_start,
        training_end=args.training_end,
        archive_ref=args.archive_ref,
    )
    print(json.dumps(result.to_dict(), indent=2))


def _verify_production_checkpoint(args: argparse.Namespace) -> None:
    config = load_production_config(args.config, project_root=args.project_root)
    result = verify_control_plane_checkpoint(
        config=config,
        manifest_path=args.manifest,
        verify_registry=not args.skip_registry,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if result.status == "failed":
        raise SystemExit(1)


def _analyze_regimes_split(args: argparse.Namespace) -> None:
    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=args.split_manifest,
        split_name="development",
        purpose="eda",
    )
    _, _, zone_summary = characterize_dataset(data, timezone=args.timezone)
    monthly, profiles, similarity = analyze_monthly_regimes(data, timezone=args.timezone)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "monthly_demand.csv", index=False)
    profiles.to_csv(output_dir / "monthly_weekly_profiles.csv", index=False)
    similarity.to_csv(output_dir / "monthly_profile_similarity.csv")
    figures = write_regime_figures(
        monthly, profiles, similarity, zone_summary, output_dir / "figures"
    )
    report = {
        "months": monthly["month"].tolist(),
        "month_count": len(monthly),
        "mean_adjacent_month_profile_correlation": float(
            similarity.to_numpy().diagonal(offset=1).mean()
        ),
        "figures": [str(path) for path in figures],
    }
    _write_json(output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


def _seasonal_gap_experiment_split(args: argparse.Namespace) -> None:
    data = load_named_split(
        monthly_root=args.monthly_root,
        manifest_path=args.split_manifest,
        split_name="development",
        purpose="development",
    )
    features = build_feature_table(data, timezone=args.timezone)
    results, predictions = run_seasonal_gap_experiment(
        features,
        target_starts=args.target_starts,
        window_days=args.window_days,
        evaluation_days=args.evaluation_days,
        gap_years=args.gap_years,
        timezone=args.timezone,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "seasonal_gap_results.csv", index=False)
    predictions.to_parquet(output_dir / "seasonal_gap_predictions.parquet", index=False)
    aggregate = (
        results.groupby("strategy")
        .agg(
            mean_mae=("mae", "mean"),
            worst_mae=("mae", "max"),
            mean_wape=("wape", "mean"),
            mean_zone_mae_p95=("zone_mae_p95", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    report = {
        "target_starts": args.target_starts,
        "window_days": args.window_days,
        "evaluation_days": args.evaluation_days,
        "gap_years": args.gap_years,
        "experiment_count": len(results),
        "aggregate": aggregate,
    }
    _write_json(output_dir / "summary.json", report)
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

    quality = subparsers.add_parser(
        "summarize-quality", help="combine monthly reports and enforce quality gates"
    )
    quality.add_argument("--report-root", required=True)
    quality.add_argument("--start-month", required=True)
    quality.add_argument("--end-month", required=True)
    quality.add_argument("--output-dir", required=True)
    quality.add_argument("--allow-missing", action="store_true")
    quality.set_defaults(handler=_summarize_quality)

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

    windows = subparsers.add_parser(
        "window-experiment-split",
        help="compare LightGBM training-history windows on fixed chronological targets",
    )
    windows.add_argument("--monthly-root", required=True)
    windows.add_argument("--split-manifest", required=True)
    windows.add_argument("--output-dir", required=True)
    windows.add_argument("--cutoffs", nargs="+", required=True)
    windows.add_argument("--windows-days", nargs="+", type=int, default=[30, 60, 90, 180, 365])
    windows.add_argument("--evaluation-days", type=int, default=28)
    windows.add_argument("--no-expanding", action="store_true")
    windows.add_argument("--timezone", default="America/New_York")
    windows.set_defaults(handler=_window_experiment_split)

    feature_ablation = subparsers.add_parser(
        "feature-ablation-split",
        help="compare feature groups on common development-period targets",
    )
    feature_ablation.add_argument("--monthly-root", required=True)
    feature_ablation.add_argument("--experiment-config", required=True)
    feature_ablation.add_argument("--output-dir", required=True)
    feature_ablation.add_argument("--training-window-days", type=int, default=365)
    feature_ablation.add_argument("--feature-sets", nargs="+")
    feature_ablation.set_defaults(handler=_feature_ablation_split)

    model_family = subparsers.add_parser(
        "model-family-experiment-split",
        help="compare production-friendly model families on common targets",
    )
    model_family.add_argument("--monthly-root", required=True)
    model_family.add_argument("--experiment-config", required=True)
    model_family.add_argument("--output-dir", required=True)
    model_family.add_argument("--training-window-days", type=int, default=365)
    model_family.add_argument("--models", nargs="+")
    model_family.set_defaults(handler=_model_family_experiment_split)

    poisson_screen = subparsers.add_parser(
        "poisson-screen-split",
        help="screen Poisson regularization and convergence on early development origins",
    )
    poisson_screen.add_argument("--monthly-root", required=True)
    poisson_screen.add_argument("--experiment-config", required=True)
    poisson_screen.add_argument("--output-dir", required=True)
    poisson_screen.add_argument("--training-window-days", type=int, default=365)
    poisson_screen.set_defaults(handler=_poisson_screen_split)

    training_history = subparsers.add_parser(
        "training-history-experiment-split",
        help="compare recent, expanding, seasonal, and hybrid histories",
    )
    training_history.add_argument("--monthly-root", required=True)
    training_history.add_argument("--experiment-config", required=True)
    training_history.add_argument("--output-dir", required=True)
    training_history.add_argument("--strategies", nargs="+")
    training_history.set_defaults(handler=_training_history_experiment_split)

    policy_validation = subparsers.add_parser(
        "validate-lifecycle-policy-split",
        help="run the pre-registered lifecycle scorecard on policy validation",
    )
    policy_validation.add_argument("--monthly-root", required=True)
    policy_validation.add_argument("--validation-config", required=True)
    policy_validation.add_argument("--output-dir", required=True)
    policy_validation.set_defaults(handler=_policy_validation_split)

    replay_features = subparsers.add_parser(
        "build-replay-feature-store",
        help="materialize month-partitioned point-in-time replay features",
    )
    replay_features.add_argument("--canonical-root", required=True)
    replay_features.add_argument("--feature-root", required=True)
    replay_features.add_argument("--start-month", required=True)
    replay_features.add_argument("--end-month", required=True)
    replay_features.add_argument("--report", required=True)
    replay_features.add_argument("--timezone", default="America/New_York")
    replay_features.add_argument("--force", action="store_true")
    replay_features.set_defaults(handler=_build_replay_feature_store)

    seed_registry = subparsers.add_parser(
        "seed-replay-registry",
        help="reconstruct the fixed pre-replay model pool without locked data",
    )
    seed_registry.add_argument("--feature-root", required=True)
    seed_registry.add_argument("--replay-config", required=True)
    seed_registry.add_argument("--output-dir", required=True)
    seed_registry.set_defaults(handler=_seed_replay_registry)

    locked_replay = subparsers.add_parser(
        "run-locked-replay",
        help="execute the frozen four-strategy replay after all preflight gates pass",
    )
    locked_replay.add_argument("--feature-root", required=True)
    locked_replay.add_argument("--replay-config", required=True)
    locked_replay.add_argument("--seed-registry", required=True)
    locked_replay.add_argument("--output-dir", required=True)
    locked_replay.add_argument(
        "--resume-completed",
        action="store_true",
        help="reuse strategies that already have a complete artifact set",
    )
    locked_replay.set_defaults(handler=_run_locked_replay)

    replay_report = subparsers.add_parser(
        "summarize-locked-replay",
        help="produce tables and figures from a completed locked replay",
    )
    replay_report.add_argument("--replay-dir", required=True)
    replay_report.add_argument("--output-dir")
    replay_report.set_defaults(handler=_summarize_locked_replay)

    replay_import = subparsers.add_parser(
        "import-locked-replay-mlflow",
        help="import the frozen replay into MLflow with explicit historical provenance",
    )
    replay_import.add_argument("--replay-dir", required=True)
    replay_import.add_argument("--tracking-uri", required=True)
    replay_import.add_argument("--experiment-name", default="nyc-taxi-locked-replay")
    replay_import.add_argument("--archive-ref", default="experiment-v1.0.0")
    replay_import.add_argument("--project-root", default=".")
    replay_import.add_argument("--registered-model-name", default="nyc-taxi-demand")
    replay_import.add_argument("--artifact-root")
    replay_import.set_defaults(handler=_import_locked_replay_mlflow)

    predict_hour = subparsers.add_parser(
        "predict-production-hour",
        help="generate one idempotent hourly batch with model and feature lineage",
    )
    predict_hour.add_argument("--feature-root", required=True)
    predict_hour.add_argument("--output-root", required=True)
    predict_hour.add_argument("--state-db", required=True)
    predict_hour.add_argument("--tracking-uri", required=True)
    predict_hour.add_argument("--model-name", default="nyc-taxi-demand")
    predict_hour.add_argument("--model-alias", default="accuracy-reference")
    predict_hour.add_argument("--target-hour", required=True)
    predict_hour.add_argument("--timezone", default="America/New_York")
    predict_hour.add_argument("--expected-zones", type=int, default=263)
    predict_hour.set_defaults(handler=_predict_production_hour)

    score_hour = subparsers.add_parser(
        "score-production-hour",
        help="join one persisted prediction batch to newly available labels",
    )
    score_hour.add_argument("--feature-root", required=True)
    score_hour.add_argument("--output-root", required=True)
    score_hour.add_argument("--state-db", required=True)
    score_hour.add_argument("--target-hour", required=True)
    score_hour.add_argument("--as-of", required=True)
    score_hour.add_argument("--timezone", default="America/New_York")
    score_hour.set_defaults(handler=_score_production_hour)

    production_hour = subparsers.add_parser(
        "run-production-hour",
        help="run the configured idempotent prediction and optional scoring path",
    )
    production_hour.add_argument("--config", default="configs/production.yaml")
    production_hour.add_argument("--project-root", default=".")
    production_hour.add_argument("--target-hour", required=True)
    production_hour.add_argument("--score-as-of")
    production_hour.set_defaults(handler=_run_production_hour)

    production_train = subparsers.add_parser(
        "train-production-model",
        help="run the configured 28-day scheduled training and registration step",
    )
    production_train.add_argument("--config", default="configs/production.yaml")
    production_train.add_argument("--project-root", default=".")
    production_train.add_argument("--as-of", required=True)
    production_train.add_argument("--force", action="store_true")
    production_train.set_defaults(handler=_train_production_model)

    production_monitoring = subparsers.add_parser(
        "build-production-monitoring",
        help="build a reproducible forecast-quality and operations dashboard",
    )
    production_monitoring.add_argument("--config", default="configs/production.yaml")
    production_monitoring.add_argument("--project-root", default=".")
    production_monitoring.add_argument("--as-of", required=True)
    production_monitoring.set_defaults(handler=_build_production_monitoring)

    dashboard = subparsers.add_parser(
        "serve-dashboard",
        help="serve the taxi-zone dashboard from registered production replay artifacts",
    )
    dashboard.add_argument("--config", default="configs/production.yaml")
    dashboard.add_argument("--replay-config", default="configs/continuous_replay.yaml")
    dashboard.add_argument("--project-root", default=".")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.set_defaults(handler=_serve_dashboard)

    dashboard_index = subparsers.add_parser(
        "build-dashboard-index",
        help="materialize fast per-zone history from registered scored artifacts",
    )
    dashboard_index.add_argument("--config", default="configs/production.yaml")
    dashboard_index.add_argument("--project-root", default=".")
    dashboard_index.set_defaults(handler=_build_dashboard_index)

    continuous = subparsers.add_parser(
        "run-continuous-replay",
        help="advance isolated historical replay and release each hour's labels after prediction",
    )
    continuous.add_argument("--config", default="configs/continuous_replay.yaml")
    continuous.add_argument("--project-root", default=".")
    continuous.add_argument("--start", required=True, help="first virtual UTC hour")
    continuous.add_argument("--end", required=True, help="exclusive virtual UTC end")
    continuous.add_argument("--interval-seconds", type=float, default=5)
    continuous.add_argument("--max-ticks", type=int)
    continuous.set_defaults(handler=_run_continuous_replay)

    production_alerts = subparsers.add_parser(
        "list-production-alerts",
        help="list durable forecast-alert incidents and their audit history",
    )
    production_alerts.add_argument("--config", default="configs/production.yaml")
    production_alerts.add_argument("--project-root", default=".")
    production_alerts.add_argument("--active-only", action="store_true")
    production_alerts.add_argument("--include-history", action="store_true")
    production_alerts.set_defaults(handler=_list_production_alerts)

    production_ack = subparsers.add_parser(
        "acknowledge-production-alert",
        help="acknowledge an active forecast-alert incident without changing model state",
    )
    production_ack.add_argument("--config", default="configs/production.yaml")
    production_ack.add_argument("--project-root", default=".")
    production_ack.add_argument("--incident-id", type=int, required=True)
    production_ack.add_argument("--actor", required=True)
    production_ack.add_argument("--note")
    production_ack.add_argument("--at", help="UTC acknowledgement time; defaults to now")
    production_ack.set_defaults(handler=_acknowledge_production_alert)

    production_audit = subparsers.add_parser(
        "audit-production-state",
        help="verify production metadata, artifacts, metrics, and registry lineage",
    )
    production_audit.add_argument("--config", default="configs/production.yaml")
    production_audit.add_argument("--project-root", default=".")
    production_audit.add_argument("--skip-registry", action="store_true")
    production_audit.set_defaults(handler=_audit_production_state)

    production_checkpoint = subparsers.add_parser(
        "checkpoint-production-state",
        help="back up SQLite state and hash every referenced production artifact",
    )
    production_checkpoint.add_argument("--config", default="configs/production.yaml")
    production_checkpoint.add_argument("--project-root", default=".")
    production_checkpoint.add_argument("--destination", required=True)
    production_checkpoint.set_defaults(handler=_checkpoint_production_state)

    production_cycle = subparsers.add_parser(
        "run-production-cycle",
        help="run one scheduler-neutral hourly production control cycle",
    )
    production_cycle.add_argument("--config", default="configs/production.yaml")
    production_cycle.add_argument("--project-root", default=".")
    production_cycle.add_argument("--as-of")
    production_cycle.add_argument("--allow-missing-previous-prediction", action="store_true")
    production_cycle.set_defaults(handler=_run_production_cycle)

    production_backfill = subparsers.add_parser(
        "backfill-production-hours",
        help="resumably predict and score a bounded interval within one model epoch",
    )
    production_backfill.add_argument("--config", default="configs/production.yaml")
    production_backfill.add_argument("--project-root", default=".")
    production_backfill.add_argument("--start", required=True)
    production_backfill.add_argument("--end", required=True)
    production_backfill.add_argument("--maximum-hours", type=int, default=744)
    production_backfill.set_defaults(handler=_backfill_production)

    historical = subparsers.add_parser(
        "extend-dashboard-history",
        help="score historical hours with the model available at each hour",
    )
    historical.add_argument("--config", default="configs/production.yaml")
    historical.add_argument("--project-root", default=".")
    historical.add_argument("--registry", default="models/history/registry.json")
    historical.add_argument("--start", required=True)
    historical.add_argument("--end", required=True)
    historical.add_argument("--maximum-hours", type=int, default=744)
    historical.add_argument("--plan-only", action="store_true")
    historical.set_defaults(handler=_extend_dashboard_history)

    production_bootstrap = subparsers.add_parser(
        "bootstrap-production-reference",
        help="idempotently import a portable native model into production MLflow",
    )
    production_bootstrap.add_argument("--config", default="configs/production.yaml")
    production_bootstrap.add_argument("--project-root", default=".")
    production_bootstrap.add_argument("--model-path", required=True)
    production_bootstrap.add_argument("--training-start", required=True)
    production_bootstrap.add_argument("--training-end", required=True)
    production_bootstrap.add_argument("--archive-ref", default="experiment-v1.0.0")
    production_bootstrap.set_defaults(handler=_bootstrap_production_reference)

    checkpoint_verify = subparsers.add_parser(
        "verify-production-checkpoint",
        help="open a checkpoint copy and verify its database, artifacts, and registry",
    )
    checkpoint_verify.add_argument("--config", default="configs/production.yaml")
    checkpoint_verify.add_argument("--project-root", default=".")
    checkpoint_verify.add_argument("--manifest", required=True)
    checkpoint_verify.add_argument("--skip-registry", action="store_true")
    checkpoint_verify.set_defaults(handler=_verify_production_checkpoint)

    regimes = subparsers.add_parser(
        "analyze-regimes-split",
        help="measure development-period monthly profiles and generate regime figures",
    )
    regimes.add_argument("--monthly-root", required=True)
    regimes.add_argument("--split-manifest", required=True)
    regimes.add_argument("--output-dir", required=True)
    regimes.add_argument("--timezone", default="America/New_York")
    regimes.set_defaults(handler=_analyze_regimes_split)

    seasonal_gap = subparsers.add_parser(
        "seasonal-gap-experiment-split",
        help="compare a season-matched historical window with the recent window",
    )
    seasonal_gap.add_argument("--monthly-root", required=True)
    seasonal_gap.add_argument("--split-manifest", required=True)
    seasonal_gap.add_argument("--output-dir", required=True)
    seasonal_gap.add_argument("--target-starts", nargs="+", required=True)
    seasonal_gap.add_argument("--window-days", type=int, default=90)
    seasonal_gap.add_argument("--evaluation-days", type=int, default=90)
    seasonal_gap.add_argument("--gap-years", type=int, default=1)
    seasonal_gap.add_argument("--timezone", default="America/New_York")
    seasonal_gap.set_defaults(handler=_seasonal_gap_experiment_split)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
