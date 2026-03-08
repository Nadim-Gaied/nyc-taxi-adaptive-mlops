"""MLflow integration for historical imports and future production runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd

from taxi_mlops.tracking.fingerprints import fingerprint_files, sha256_file


@dataclass(frozen=True)
class HistoricalImportResult:
    """Stable identifiers produced when a completed replay is imported."""

    parent_run_id: str
    strategy_run_ids: dict[str, str]
    registered_model_name: str | None
    registered_model_version: str | None
    reused_existing_import: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            'MLflow is optional. Install it with: pip install -e ".[tracking]"'
        ) from error
    return mlflow


def _numeric_metrics(row: pd.Series) -> dict[str, float]:
    excluded = {"strategy"}
    metrics: dict[str, float] = {}
    for name, value in row.items():
        if name in excluded or pd.isna(value) or isinstance(value, bool):
            continue
        if isinstance(value, Number):
            metrics[str(name)] = float(value)
    return metrics


def _existing_import(client: Any, experiment_id: str, archive_ref: str) -> Any | None:
    escaped = archive_ref.replace("'", "\\'")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.import_kind = 'locked_replay_summary' AND tags.archive_ref = '{escaped}'"
        ),
        max_results=2,
    )
    if len(runs) > 1:
        raise RuntimeError(f"multiple historical imports exist for {archive_ref}")
    return runs[0] if runs else None


def _child_strategy_runs(client: Any, experiment_id: str, parent_run_id: str) -> dict[str, str]:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=100,
    )
    return {
        run.data.tags["strategy"]: run.info.run_id for run in runs if "strategy" in run.data.tags
    }


def _log_existing_artifacts(mlflow: Any, paths: list[Path], artifact_path: str) -> None:
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _register_fixed_reference_model(
    mlflow: Any,
    *,
    replay_dir: Path,
    registered_model_name: str,
    archive_ref: str,
) -> str:
    import lightgbm as lgb

    strategy_dir = replay_dir / "fixed_periodic_28d"
    registry_path = strategy_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    model_id = registry.get("active_model_id")
    if not model_id:
        raise ValueError("fixed-periodic registry has no active model")
    record = next(
        (item for item in registry["models"] if item["model_id"] == model_id),
        None,
    )
    if record is None:
        raise ValueError(f"active model is missing from registry: {model_id}")
    model_path = Path(record["artifact_path"])
    if not model_path.is_absolute():
        direct = model_path.resolve()
        model_path = direct if direct.is_file() else strategy_dir / "models" / f"{model_id}.txt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    booster = lgb.Booster(model_file=str(model_path))
    model_info = mlflow.lightgbm.log_model(
        booster,
        name="model",
        registered_model_name=registered_model_name,
        metadata={
            "archive_ref": archive_ref,
            "source_model_id": model_id,
            "model_sha256": sha256_file(model_path),
        },
    )
    version = str(model_info.registered_model_version)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(registered_model_name, version, "archive_ref", archive_ref)
    client.set_model_version_tag(registered_model_name, version, "source_model_id", model_id)
    client.set_model_version_tag(
        registered_model_name, version, "training_start", str(record["training_start"])
    )
    client.set_model_version_tag(
        registered_model_name, version, "training_end", str(record["training_end"])
    )
    client.set_model_version_tag(
        registered_model_name, version, "history_strategy", str(record["history_strategy"])
    )
    client.set_model_version_tag(
        registered_model_name, version, "feature_set", str(record["feature_set"])
    )
    client.set_registered_model_alias(registered_model_name, "accuracy-reference", version)
    return version


def import_locked_replay_to_mlflow(
    *,
    replay_dir: str | Path,
    tracking_uri: str,
    experiment_name: str,
    archive_ref: str,
    project_root: str | Path,
    registered_model_name: str | None = "nyc-taxi-demand",
    artifact_root: str | None = None,
) -> HistoricalImportResult:
    """Import the completed replay once and preserve its historical provenance.

    The import is idempotent per ``archive_ref``. It records the replay as
    historical evidence and never claims that MLflow was present during the run.
    """

    mlflow = _require_mlflow()
    replay_dir = Path(replay_dir).resolve()
    project_root = Path(project_root).resolve()
    comparison_path = replay_dir / "strategy_comparison.csv"
    summary_path = replay_dir / "summary.json"
    comparison = pd.read_csv(comparison_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if comparison.empty or comparison["observations"].nunique() != 1:
        raise ValueError("locked replay strategies do not share one target set")

    lineage_paths = [
        project_root / "configs" / "splits.yaml",
        project_root / "configs" / "development_experiments.yaml",
        project_root / "configs" / "lifecycle_policy_final_v1.yaml",
        project_root / "configs" / "locked_replay_experiment_v1.yaml",
        comparison_path,
        summary_path,
    ]
    lineage_fingerprint = fingerprint_files(lineage_paths, root=project_root)

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = artifact_root
        if artifact_location and "://" not in artifact_location:
            artifact_location = Path(artifact_location).resolve().as_uri()
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(experiment_name)
    existing = _existing_import(client, experiment.experiment_id, archive_ref)
    if existing is not None:
        return HistoricalImportResult(
            parent_run_id=existing.info.run_id,
            strategy_run_ids=_child_strategy_runs(
                client, experiment.experiment_id, existing.info.run_id
            ),
            registered_model_name=existing.data.tags.get("registered_model_name"),
            registered_model_version=existing.data.tags.get("registered_model_version"),
            reused_existing_import=True,
        )

    strategy_run_ids: dict[str, str] = {}
    registered_version: str | None = None
    with mlflow.start_run(
        run_name=f"locked-replay-{archive_ref}",
        tags={
            "import_kind": "locked_replay_summary",
            "archive_ref": archive_ref,
            "historical_import": "true",
            "locked_protocol_changed": str(summary["locked_protocol_changed"]).lower(),
            "lineage_fingerprint": lineage_fingerprint,
        },
    ) as parent:
        mlflow.log_params(
            {
                "replay_start": "2020-01-01T00:00:00-05:00",
                "replay_end": "2026-01-01T00:00:00-05:00",
                "strategy_count": len(comparison),
                "observations_per_strategy": int(comparison["observations"].iloc[0]),
                "best_overall_wape_strategy": summary["best_overall_wape_strategy"],
            }
        )
        _log_existing_artifacts(
            mlflow,
            [comparison_path, summary_path, *lineage_paths[:4]],
            "frozen_reference",
        )
        analysis_dir = replay_dir / "analysis"
        if analysis_dir.is_dir():
            mlflow.log_artifacts(str(analysis_dir), artifact_path="analysis")

        for row in comparison.itertuples(index=False):
            strategy = str(row.strategy)
            strategy_dir = replay_dir / strategy
            with mlflow.start_run(
                run_name=strategy,
                nested=True,
                tags={
                    "strategy": strategy,
                    "archive_ref": archive_ref,
                    "historical_import": "true",
                },
            ) as child:
                strategy_run_ids[strategy] = child.info.run_id
                mlflow.log_metrics(_numeric_metrics(pd.Series(row._asdict())))
                mlflow.log_param("feature_set", "full_v1")
                mlflow.log_param("model_family", "lightgbm_poisson")
                _log_existing_artifacts(
                    mlflow,
                    [
                        strategy_dir / "summary.json",
                        strategy_dir / "weekly_metrics.csv",
                        strategy_dir / "tier_metrics.csv",
                        strategy_dir / "zone_metrics.csv",
                        strategy_dir / "lifecycle_audit.csv",
                        strategy_dir / "registry.json",
                    ],
                    "replay_outputs",
                )
                if strategy == "fixed_periodic_28d" and registered_model_name:
                    registered_version = _register_fixed_reference_model(
                        mlflow,
                        replay_dir=replay_dir,
                        registered_model_name=registered_model_name,
                        archive_ref=archive_ref,
                    )

        if registered_model_name and registered_version:
            mlflow.set_tag("registered_model_name", registered_model_name)
            mlflow.set_tag("registered_model_version", registered_version)

        return HistoricalImportResult(
            parent_run_id=parent.info.run_id,
            strategy_run_ids=strategy_run_ids,
            registered_model_name=registered_model_name if registered_version else None,
            registered_model_version=registered_version,
            reused_existing_import=False,
        )
