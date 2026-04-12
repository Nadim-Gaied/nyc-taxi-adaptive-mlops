"""Common-target comparison of production-friendly forecasting models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from taxi_mlops.evaluation import (
    assign_demand_tiers,
    regression_metrics,
    tier_regression_metrics,
    validate_common_targets,
)
from taxi_mlops.models.window_experiment import DEFAULT_MODEL_PARAMETERS

SUPPORTED_MODELS = (
    "daily_weekly_blend",
    "ridge",
    "poisson_regression",
    "lightgbm_poisson",
)
LINEAR_CATEGORICAL_COLUMNS = [
    "pickup_zone_id",
    "hour_of_day",
    "day_of_week",
    "month",
]


def _linear_pipeline(
    feature_columns: Sequence[str], model_name: str, parameters: Mapping[str, object]
) -> Pipeline:
    categorical = [column for column in LINEAR_CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    transformer = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        sparse_threshold=1.0,
    )
    if model_name == "ridge":
        defaults: dict[str, object] = {"alpha": 1.0, "solver": "lsqr", "tol": 1e-4}
        estimator = Ridge(**{**defaults, **parameters})
    elif model_name == "poisson_regression":
        defaults = {
            "alpha": 1.0,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 1e-4,
        }
        estimator = PoissonRegressor(**{**defaults, **parameters})
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unsupported linear model: {model_name}")
    return Pipeline([("features", transformer), ("model", estimator)])


def _fit_predict(
    model_name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    parameters: Mapping[str, object],
) -> tuple[np.ndarray, float, float, dict[str, object]]:
    if model_name == "lightgbm_poisson":
        x_train = train[feature_columns].copy()
        x_evaluation = evaluation[feature_columns].copy()
        categories = sorted(train["pickup_zone_id"].unique())
        x_train["pickup_zone_id"] = pd.Categorical(x_train["pickup_zone_id"], categories=categories)
        x_evaluation["pickup_zone_id"] = pd.Categorical(
            x_evaluation["pickup_zone_id"], categories=categories
        )
        estimator = lgb.LGBMRegressor(**{**DEFAULT_MODEL_PARAMETERS, **parameters})
        started = perf_counter()
        estimator.fit(
            x_train,
            train["pickup_count"],
            categorical_feature=["pickup_zone_id"],
        )
        training_seconds = perf_counter() - started
        started = perf_counter()
        prediction = estimator.predict(x_evaluation)
        inference_seconds = perf_counter() - started
        diagnostics: dict[str, object] = {
            "iterations": None,
            "hit_iteration_limit": False,
        }
    else:
        estimator = _linear_pipeline(feature_columns, model_name, parameters)
        started = perf_counter()
        estimator.fit(train[feature_columns], train["pickup_count"])
        training_seconds = perf_counter() - started
        started = perf_counter()
        prediction = estimator.predict(evaluation[feature_columns])
        inference_seconds = perf_counter() - started
        fitted_model = estimator.named_steps["model"]
        iterations = (
            getattr(fitted_model, "n_iter_", None) if model_name == "poisson_regression" else None
        )
        configured_limit = int(parameters.get("max_iter", 100))
        diagnostics = {
            "iterations": int(iterations) if iterations is not None else None,
            "hit_iteration_limit": bool(
                model_name == "poisson_regression"
                and iterations is not None
                and int(iterations) >= configured_limit
            ),
        }
    return (
        np.clip(prediction, 0.0, None),
        training_seconds,
        inference_seconds,
        diagnostics,
    )


def run_model_family_experiment(
    feature_table: pd.DataFrame,
    *,
    cutoffs: list[str | pd.Timestamp],
    feature_columns: Sequence[str],
    model_names: Sequence[str] = SUPPORTED_MODELS,
    training_window_days: int = 365,
    evaluation_days: int = 28,
    timezone: str = "America/New_York",
    tier_boundaries: Sequence[float] = (0.80, 0.95),
    tier_labels: Sequence[str] = ("high", "medium", "low"),
    model_parameters: Mapping[str, Mapping[str, object]] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare model families using identical histories, features, and targets."""

    unknown = sorted(set(model_names).difference(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"unsupported model names: {', '.join(unknown)}")
    if not model_names:
        raise ValueError("at least one model name is required")
    if training_window_days <= 0 or evaluation_days <= 0:
        raise ValueError("training and evaluation lengths must be positive")

    columns = list(feature_columns)
    required = {
        "pickup_hour",
        "pickup_zone_id",
        "pickup_count",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        *columns,
    }
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {', '.join(sorted(missing))}")

    data = feature_table.copy()
    data["pickup_hour"] = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
    data = data.sort_values(["pickup_hour", "pickup_zone_id"])
    parameters = {name: dict(values) for name, values in (model_parameters or {}).items()}
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        signature_payload = {
            "cutoffs": [str(value) for value in cutoffs],
            "feature_columns": columns,
            "model_names": list(model_names),
            "training_window_days": training_window_days,
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
            raise ValueError("model-family checkpoint does not match this experiment")
        signature_path.write_text(signature, encoding="utf-8")

    complete_baseline: pd.DataFrame | None = None
    if "daily_weekly_blend" in model_names:
        complete_baseline = data[
            [
                "pickup_hour",
                "pickup_zone_id",
                "pickup_count",
                "lag_1h",
                "lag_24h",
                "lag_168h",
            ]
        ].copy()
        both_available = (
            complete_baseline["lag_24h"].notna() & complete_baseline["lag_168h"].notna()
        )
        complete_baseline["prediction"] = np.where(
            both_available,
            0.5 * complete_baseline["lag_24h"] + 0.5 * complete_baseline["lag_168h"],
            complete_baseline["lag_168h"]
            .fillna(complete_baseline["lag_24h"])
            .fillna(complete_baseline["lag_1h"]),
        )
    result_rows: list[dict[str, object]] = []
    tier_frames: list[pd.DataFrame] = []
    for raw_cutoff in cutoffs:
        cutoff = pd.Timestamp(raw_cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("cutoffs must include an explicit UTC offset")
        cutoff = cutoff.tz_convert("UTC")
        cutoff_local = cutoff.tz_convert(timezone)
        evaluation_end = (cutoff_local + pd.DateOffset(days=evaluation_days)).tz_convert("UTC")
        training_start = (cutoff_local - pd.DateOffset(days=training_window_days)).tz_convert("UTC")
        train = data.loc[data["pickup_hour"].ge(training_start) & data["pickup_hour"].lt(cutoff)]
        evaluation = data.loc[
            data["pickup_hour"].ge(cutoff) & data["pickup_hour"].lt(evaluation_end)
        ]
        canonical_targets = evaluation[["pickup_hour", "pickup_zone_id", "pickup_count"]]
        tiers = assign_demand_tiers(
            train,
            cumulative_boundaries=tier_boundaries,
            tier_labels=tier_labels,
        )

        for model_name in model_names:
            checkpoint_stem = f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}_{model_name}"
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
            if (
                result_checkpoint is not None
                and tier_checkpoint is not None
                and result_checkpoint.exists()
                and tier_checkpoint.exists()
            ):
                result_rows.append(json.loads(result_checkpoint.read_text(encoding="utf-8")))
                tier_frames.append(pd.read_csv(tier_checkpoint))
                print(f"Resumed {model_name} at {cutoff}", flush=True)
                continue

            if model_name == "daily_weekly_blend":
                if complete_baseline is None:  # pragma: no cover - guarded above
                    raise RuntimeError("complete baseline was not prepared")
                predictions = complete_baseline.loc[
                    complete_baseline["pickup_hour"].ge(cutoff)
                    & complete_baseline["pickup_hour"].lt(evaluation_end),
                    ["pickup_hour", "pickup_zone_id", "pickup_count", "prediction"],
                ].copy()
                training_seconds = 0.0
                inference_seconds = 0.0
                diagnostics = {"iterations": None, "hit_iteration_limit": False}
            else:
                prediction, training_seconds, inference_seconds, diagnostics = _fit_predict(
                    model_name,
                    train,
                    evaluation,
                    columns,
                    parameters.get(model_name, {}),
                )
                predictions = canonical_targets.copy()
                predictions["prediction"] = prediction

            validate_common_targets({"reference": canonical_targets, model_name: predictions})
            metrics = regression_metrics(predictions)
            result = {
                "cutoff": str(cutoff),
                "model_name": model_name,
                "training_start": str(training_start),
                "training_end": str(cutoff),
                "evaluation_start": str(cutoff),
                "evaluation_end": str(evaluation_end),
                "training_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "feature_count": len(columns),
                "training_seconds": float(training_seconds),
                "inference_seconds": float(inference_seconds),
                **diagnostics,
                **metrics,
            }
            tier_metrics = tier_regression_metrics(predictions, tiers)
            tier_metrics["cutoff"] = str(cutoff)
            tier_metrics["model_name"] = model_name
            result_rows.append(result)
            tier_frames.append(tier_metrics)
            if result_checkpoint is not None and tier_checkpoint is not None:
                result_checkpoint.write_text(
                    json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
                )
                tier_metrics.to_csv(tier_checkpoint, index=False)
            print(f"Completed {model_name} at {cutoff}", flush=True)

    results = pd.DataFrame(result_rows).sort_values(["cutoff", "wape", "model_name"])
    tier_results = pd.concat(tier_frames, ignore_index=True).sort_values(
        ["cutoff", "model_name", "demand_tier"]
    )
    return results.reset_index(drop=True), tier_results.reset_index(drop=True)
