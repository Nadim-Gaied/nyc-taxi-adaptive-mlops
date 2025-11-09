"""Controlled LightGBM training-history window experiments."""

from __future__ import annotations

from time import perf_counter

import lightgbm as lgb
import pandas as pd

from taxi_mlops.evaluation import regression_metrics

DEFAULT_FEATURE_COLUMNS = [
    "pickup_zone_id",
    "hour_of_day",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
    "rolling_std_168h",
]

DEFAULT_MODEL_PARAMETERS: dict[str, object] = {
    "objective": "poisson",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 1.0,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def _as_utc(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


def _model_matrices(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x_train = train[feature_columns].copy()
    x_evaluation = evaluation[feature_columns].copy()
    zone_categories = sorted(train["pickup_zone_id"].unique())
    x_train["pickup_zone_id"] = pd.Categorical(
        x_train["pickup_zone_id"], categories=zone_categories
    )
    x_evaluation["pickup_zone_id"] = pd.Categorical(
        x_evaluation["pickup_zone_id"], categories=zone_categories
    )
    return x_train, train["pickup_count"], x_evaluation


def run_training_window_experiment(
    feature_table: pd.DataFrame,
    *,
    cutoffs: list[str | pd.Timestamp],
    windows_days: tuple[int, ...] = (30, 60, 90, 180, 365),
    include_expanding: bool = True,
    evaluation_days: int = 28,
    feature_columns: list[str] | None = None,
    model_parameters: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vary only training history while holding model, features, and targets fixed."""

    if not cutoffs:
        raise ValueError("at least one cutoff is required")
    if any(window <= 0 for window in windows_days):
        raise ValueError("training windows must be positive")
    if evaluation_days <= 0:
        raise ValueError("evaluation_days must be positive")

    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    required = {"pickup_hour", "pickup_zone_id", "pickup_count", *feature_columns}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {', '.join(sorted(missing))}")

    data = feature_table.copy()
    data["pickup_hour"] = pd.to_datetime(data["pickup_hour"], utc=True, errors="raise")
    data = data.sort_values(["pickup_hour", "pickup_zone_id"])
    parameters = {**DEFAULT_MODEL_PARAMETERS, **(model_parameters or {})}
    window_options: list[int | None] = list(windows_days)
    if include_expanding:
        window_options.append(None)

    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for raw_cutoff in cutoffs:
        cutoff = _as_utc(raw_cutoff, "cutoff")
        evaluation_end = cutoff + pd.Timedelta(days=evaluation_days)
        evaluation = data.loc[
            data["pickup_hour"].ge(cutoff) & data["pickup_hour"].lt(evaluation_end)
        ].dropna(subset=feature_columns)
        if evaluation.empty:
            raise ValueError(f"no evaluation rows for cutoff {cutoff}")

        expected_evaluation_keys = evaluation[["pickup_hour", "pickup_zone_id"]]
        for window_days in window_options:
            training_start = (
                data["pickup_hour"].min()
                if window_days is None
                else cutoff - pd.Timedelta(days=window_days)
            )
            train = data.loc[
                data["pickup_hour"].ge(training_start) & data["pickup_hour"].lt(cutoff)
            ].dropna(subset=feature_columns)
            if train.empty:
                raise ValueError(
                    f"no training rows for cutoff {cutoff} and window {window_days}"
                )
            if not expected_evaluation_keys.equals(
                evaluation[["pickup_hour", "pickup_zone_id"]]
            ):
                raise RuntimeError("evaluation targets changed between training windows")

            x_train, y_train, x_evaluation = _model_matrices(
                train, evaluation, feature_columns
            )
            model = lgb.LGBMRegressor(**parameters)
            started = perf_counter()
            model.fit(
                x_train,
                y_train,
                categorical_feature=["pickup_zone_id"],
            )
            training_seconds = perf_counter() - started

            predictions = evaluation[["pickup_hour", "pickup_zone_id", "pickup_count"]].copy()
            predictions["prediction"] = model.predict(x_evaluation).clip(0.0)
            window_label = "expanding" if window_days is None else f"{window_days}d"
            predictions["cutoff"] = cutoff
            predictions["training_window"] = window_label
            metrics = regression_metrics(predictions)
            result_rows.append(
                {
                    "cutoff": str(cutoff),
                    "training_window": window_label,
                    "training_start": str(train["pickup_hour"].min()),
                    "training_end": str(train["pickup_hour"].max()),
                    "evaluation_start": str(evaluation["pickup_hour"].min()),
                    "evaluation_end": str(evaluation["pickup_hour"].max()),
                    "training_rows": len(train),
                    "evaluation_rows": len(evaluation),
                    "training_seconds": training_seconds,
                    **metrics,
                }
            )
            prediction_frames.append(predictions)

    results = pd.DataFrame(result_rows).sort_values(["cutoff", "mae", "training_window"])
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    return results.reset_index(drop=True), all_predictions
