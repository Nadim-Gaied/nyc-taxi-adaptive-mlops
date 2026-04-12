"""Common-target comparison of recent, expanding, and seasonal histories."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import pandas as pd

from taxi_mlops.evaluation import (
    assign_demand_tiers,
    regression_metrics,
    tier_regression_metrics,
    validate_common_targets,
)
from taxi_mlops.models.window_experiment import (
    DEFAULT_MODEL_PARAMETERS,
    _model_matrices,
)

SUPPORTED_HISTORIES = (
    "recent_90d",
    "recent_180d",
    "recent_365d",
    "expanding",
    "seasonal_90d_1y_gap",
    "recent_90d_plus_seasonal_90d",
)


def _history_intervals(
    strategy: str,
    cutoff: pd.Timestamp,
    data_start: pd.Timestamp,
    timezone: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    local_cutoff = cutoff.tz_convert(timezone)
    if strategy == "expanding":
        return [(data_start, cutoff)]
    if strategy == "recent_90d_plus_seasonal_90d":
        seasonal_start = (local_cutoff - pd.DateOffset(years=1)).tz_convert("UTC")
        seasonal_end = (seasonal_start.tz_convert(timezone) + pd.DateOffset(days=90)).tz_convert(
            "UTC"
        )
        recent_start = (local_cutoff - pd.DateOffset(days=90)).tz_convert("UTC")
        return [(seasonal_start, seasonal_end), (recent_start, cutoff)]
    if strategy.startswith("recent_") and strategy.endswith("d"):
        days = int(strategy.removeprefix("recent_").removesuffix("d"))
        start = (local_cutoff - pd.DateOffset(days=days)).tz_convert("UTC")
        return [(start, cutoff)]
    seasonal_start = (local_cutoff - pd.DateOffset(years=1)).tz_convert("UTC")
    seasonal_end = (seasonal_start.tz_convert(timezone) + pd.DateOffset(days=90)).tz_convert("UTC")
    if strategy == "seasonal_90d_1y_gap":
        return [(seasonal_start, seasonal_end)]
    raise ValueError(f"unsupported training history: {strategy}")


def _select_intervals(
    data: pd.DataFrame, intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    selected = pd.Series(False, index=data.index)
    for start, end in intervals:
        selected |= data["pickup_hour"].ge(start) & data["pickup_hour"].lt(end)
    return data.loc[selected].copy()


def _seven_day_metrics(
    predictions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    timezone: str,
) -> pd.DataFrame:
    local_dates = predictions["pickup_hour"].dt.tz_convert(timezone).dt.tz_localize(None)
    cutoff_local = cutoff.tz_convert(timezone).tz_localize(None)
    block = ((local_dates.dt.normalize() - cutoff_local.normalize()).dt.days // 7).astype("int8")
    rows: list[dict[str, object]] = []
    for block_index, frame in predictions.groupby(block, sort=True):
        metrics = regression_metrics(frame)
        rows.append(
            {
                "week_index": int(block_index) + 1,
                "week_start": str((cutoff_local + pd.DateOffset(days=int(block_index) * 7)).date()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_training_history_experiment(
    feature_table: pd.DataFrame,
    *,
    cutoffs: list[str | pd.Timestamp],
    feature_columns: Sequence[str],
    strategies: Sequence[str] = SUPPORTED_HISTORIES,
    evaluation_days: int = 28,
    timezone: str = "America/New_York",
    tier_boundaries: Sequence[float] = (0.80, 0.95),
    tier_labels: Sequence[str] = ("high", "medium", "low"),
    model_parameters: Mapping[str, object] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare training histories with one fixed model, feature set, and target grid."""

    unknown = sorted(set(strategies).difference(SUPPORTED_HISTORIES))
    if unknown:
        raise ValueError(f"unsupported training histories: {', '.join(unknown)}")
    if not strategies:
        raise ValueError("at least one training history is required")
    if evaluation_days <= 0:
        raise ValueError("evaluation_days must be positive")

    columns = list(feature_columns)
    required = {"pickup_hour", "pickup_zone_id", "pickup_count", *columns}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {', '.join(sorted(missing))}")

    data = feature_table.copy()
    data["pickup_hour"] = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
    data = data.sort_values(["pickup_hour", "pickup_zone_id"])
    parameters = {**DEFAULT_MODEL_PARAMETERS, **(model_parameters or {})}
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        signature_payload = {
            "cutoffs": [str(value) for value in cutoffs],
            "feature_columns": columns,
            "strategies": list(strategies),
            "evaluation_days": evaluation_days,
            "timezone": timezone,
            "tier_boundaries": list(tier_boundaries),
            "tier_labels": list(tier_labels),
            "model_parameters": parameters,
        }
        signature = sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        signature_path = checkpoint_root / "signature.txt"
        if signature_path.exists() and signature_path.read_text(encoding="utf-8") != signature:
            raise ValueError("training-history checkpoint does not match this experiment")
        signature_path.write_text(signature, encoding="utf-8")

    result_rows: list[dict[str, object]] = []
    tier_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    data_start = pd.Timestamp(data["pickup_hour"].min())
    for raw_cutoff in cutoffs:
        cutoff = pd.Timestamp(raw_cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("cutoffs must include an explicit UTC offset")
        cutoff = cutoff.tz_convert("UTC")
        evaluation_end = (
            cutoff.tz_convert(timezone) + pd.DateOffset(days=evaluation_days)
        ).tz_convert("UTC")
        evaluation = data.loc[
            data["pickup_hour"].ge(cutoff) & data["pickup_hour"].lt(evaluation_end)
        ].copy()
        canonical_targets = evaluation[["pickup_hour", "pickup_zone_id", "pickup_count"]]
        tier_reference_start = (cutoff.tz_convert(timezone) - pd.DateOffset(days=365)).tz_convert(
            "UTC"
        )
        tier_reference = data.loc[
            data["pickup_hour"].ge(tier_reference_start) & data["pickup_hour"].lt(cutoff)
        ]
        tiers = assign_demand_tiers(
            tier_reference,
            cumulative_boundaries=tier_boundaries,
            tier_labels=tier_labels,
        )

        for strategy in strategies:
            checkpoint_stem = f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}_{strategy}"
            result_checkpoint = (
                checkpoint_root / f"{checkpoint_stem}_result.json"
                if checkpoint_root is not None
                else None
            )
            tier_checkpoint = (
                checkpoint_root / f"{checkpoint_stem}_tiers.csv"
                if checkpoint_root is not None
                else None
            )
            weekly_checkpoint = (
                checkpoint_root / f"{checkpoint_stem}_weeks.csv"
                if checkpoint_root is not None
                else None
            )
            if (
                result_checkpoint is not None
                and tier_checkpoint is not None
                and weekly_checkpoint is not None
                and result_checkpoint.exists()
                and tier_checkpoint.exists()
                and weekly_checkpoint.exists()
            ):
                result_rows.append(json.loads(result_checkpoint.read_text(encoding="utf-8")))
                tier_frames.append(pd.read_csv(tier_checkpoint))
                weekly_frames.append(pd.read_csv(weekly_checkpoint))
                print(f"Resumed {strategy} at {cutoff}", flush=True)
                continue

            intervals = _history_intervals(strategy, cutoff, data_start, timezone)
            train = _select_intervals(data, intervals)
            if train.empty:
                raise ValueError(f"no training rows for {strategy} at {cutoff}")
            x_train, y_train, x_evaluation = _model_matrices(train, evaluation, columns)
            model = lgb.LGBMRegressor(**parameters)
            started = perf_counter()
            model.fit(x_train, y_train, categorical_feature=["pickup_zone_id"])
            training_seconds = perf_counter() - started
            started = perf_counter()
            prediction = model.predict(x_evaluation).clip(0.0)
            inference_seconds = perf_counter() - started
            predictions = canonical_targets.copy()
            predictions["prediction"] = prediction
            validate_common_targets({"reference": canonical_targets, strategy: predictions})

            metrics = regression_metrics(predictions)
            result = {
                "cutoff": str(cutoff),
                "training_history": strategy,
                "training_intervals": json.dumps(
                    [[str(start), str(end)] for start, end in intervals]
                ),
                "training_rows": int(len(train)),
                "evaluation_start": str(cutoff),
                "evaluation_end": str(evaluation_end),
                "evaluation_rows": int(len(evaluation)),
                "training_seconds": float(training_seconds),
                "inference_seconds": float(inference_seconds),
                **metrics,
            }
            tier_metrics = tier_regression_metrics(predictions, tiers)
            tier_metrics["cutoff"] = str(cutoff)
            tier_metrics["training_history"] = strategy
            weekly_metrics = _seven_day_metrics(predictions, cutoff=cutoff, timezone=timezone)
            weekly_metrics["cutoff"] = str(cutoff)
            weekly_metrics["training_history"] = strategy
            result_rows.append(result)
            tier_frames.append(tier_metrics)
            weekly_frames.append(weekly_metrics)
            if (
                result_checkpoint is not None
                and tier_checkpoint is not None
                and weekly_checkpoint is not None
            ):
                result_checkpoint.write_text(
                    json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
                )
                tier_metrics.to_csv(tier_checkpoint, index=False)
                weekly_metrics.to_csv(weekly_checkpoint, index=False)
            print(f"Completed {strategy} at {cutoff}", flush=True)

    results = pd.DataFrame(result_rows).sort_values(["cutoff", "wape", "training_history"])
    tiers = pd.concat(tier_frames, ignore_index=True).sort_values(
        ["cutoff", "training_history", "demand_tier"]
    )
    weeks = pd.concat(weekly_frames, ignore_index=True).sort_values(
        ["cutoff", "training_history", "week_index"]
    )
    return (
        results.reset_index(drop=True),
        tiers.reset_index(drop=True),
        weeks.reset_index(drop=True),
    )
