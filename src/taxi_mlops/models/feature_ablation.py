"""Controlled feature-group ablation on common chronological targets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import pandas as pd

from taxi_mlops.evaluation import (
    assign_demand_tiers,
    tier_regression_metrics,
    validate_common_targets,
)
from taxi_mlops.models.window_experiment import run_training_window_experiment


def run_feature_ablation_experiment(
    feature_table: pd.DataFrame,
    *,
    cutoffs: list[str | pd.Timestamp],
    feature_sets: Mapping[str, Sequence[str]],
    training_window_days: int = 365,
    evaluation_days: int = 28,
    timezone: str = "America/New_York",
    tier_boundaries: Sequence[float] = (0.80, 0.95),
    tier_labels: Sequence[str] = ("high", "medium", "low"),
    model_parameters: dict[str, object] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate feature sets while fixing model, history, and target rows."""

    if not feature_sets:
        raise ValueError("at least one feature set is required")
    if training_window_days <= 0:
        raise ValueError("training_window_days must be positive")

    data = feature_table.copy()
    data["pickup_hour"] = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
    result_frames: list[pd.DataFrame] = []
    tier_frames: list[pd.DataFrame] = []
    targets_by_cutoff: dict[pd.Timestamp, pd.DataFrame] = {}
    for raw_cutoff in cutoffs:
        cutoff = pd.Timestamp(raw_cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("cutoffs must include an explicit UTC offset")
        cutoff = cutoff.tz_convert("UTC")
        evaluation_end = (
            cutoff.tz_convert(timezone) + pd.DateOffset(days=evaluation_days)
        ).tz_convert("UTC")
        targets_by_cutoff[cutoff] = data.loc[
            data["pickup_hour"].ge(cutoff)
            & data["pickup_hour"].lt(evaluation_end),
            ["pickup_hour", "pickup_zone_id", "pickup_count"],
        ]

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        signature_payload = {
            "cutoffs": [str(value) for value in cutoffs],
            "feature_sets": {name: list(columns) for name, columns in feature_sets.items()},
            "training_window_days": training_window_days,
            "evaluation_days": evaluation_days,
            "timezone": timezone,
            "tier_boundaries": list(tier_boundaries),
            "tier_labels": list(tier_labels),
            "model_parameters": model_parameters,
        }
        signature = sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        signature_path = checkpoint_root / "signature.txt"
        if signature_path.exists() and signature_path.read_text(encoding="utf-8") != signature:
            raise ValueError("feature-ablation checkpoint does not match this experiment")
        signature_path.write_text(signature, encoding="utf-8")

    for feature_set_name, raw_columns in feature_sets.items():
        feature_columns = list(raw_columns)
        if not feature_columns:
            raise ValueError(f"feature set {feature_set_name} contains no columns")
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError(f"feature set {feature_set_name} contains duplicate columns")

        result_checkpoint = (
            checkpoint_root / f"{feature_set_name}_results.csv"
            if checkpoint_root is not None
            else None
        )
        tier_checkpoint = (
            checkpoint_root / f"{feature_set_name}_tiers.csv"
            if checkpoint_root is not None
            else None
        )
        if (
            result_checkpoint is not None
            and tier_checkpoint is not None
            and result_checkpoint.exists()
            and tier_checkpoint.exists()
        ):
            result_frames.append(pd.read_csv(result_checkpoint))
            tier_frames.append(pd.read_csv(tier_checkpoint))
            print(f"Resumed completed feature set: {feature_set_name}")
            continue

        results, predictions = run_training_window_experiment(
            data,
            cutoffs=cutoffs,
            windows_days=(training_window_days,),
            include_expanding=False,
            evaluation_days=evaluation_days,
            timezone=timezone,
            feature_columns=feature_columns,
            model_parameters=model_parameters,
        )
        results["feature_set"] = feature_set_name
        results["feature_count"] = len(feature_columns)
        feature_tier_frames: list[pd.DataFrame] = []

        for cutoff, candidate_predictions in predictions.groupby("cutoff", sort=True):
            cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
            validate_common_targets(
                {
                    "reference": targets_by_cutoff[cutoff],
                    feature_set_name: candidate_predictions,
                }
            )

            cutoff_local = cutoff.tz_convert(timezone)
            training_start = (
                cutoff_local - pd.DateOffset(days=training_window_days)
            ).tz_convert("UTC")
            training = data.loc[
                data["pickup_hour"].ge(training_start)
                & data["pickup_hour"].lt(cutoff)
            ]
            tiers = assign_demand_tiers(
                training,
                cumulative_boundaries=tier_boundaries,
                tier_labels=tier_labels,
            )
            tier_metrics = tier_regression_metrics(candidate_predictions, tiers)
            tier_metrics["cutoff"] = cutoff
            tier_metrics["feature_set"] = feature_set_name
            feature_tier_frames.append(tier_metrics)

        feature_tiers = pd.concat(feature_tier_frames, ignore_index=True)
        result_frames.append(results)
        tier_frames.append(feature_tiers)
        if result_checkpoint is not None and tier_checkpoint is not None:
            results.to_csv(result_checkpoint, index=False)
            feature_tiers.to_csv(tier_checkpoint, index=False)
            print(f"Completed feature set: {feature_set_name}")

    all_results = pd.concat(result_frames, ignore_index=True).sort_values(
        ["cutoff", "wape", "feature_set"]
    )
    all_tiers = pd.concat(tier_frames, ignore_index=True).sort_values(
        ["cutoff", "feature_set", "demand_tier"]
    )
    return all_results.reset_index(drop=True), all_tiers.reset_index(drop=True)
