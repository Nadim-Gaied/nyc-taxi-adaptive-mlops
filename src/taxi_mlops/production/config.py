"""Validated production configuration and repository-relative path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrackingConfig:
    uri: str
    artifact_root: Path
    experiment_name: str
    registered_model_name: str
    production_alias: str


@dataclass(frozen=True)
class StorageConfig:
    feature_root: Path
    prediction_root: Path
    score_root: Path
    state_db: Path


@dataclass(frozen=True)
class LifecycleConfig:
    strategy: str
    training_history: str
    retrain_every_calendar_days: int
    statistical_promotion_gate: bool
    technical_validation_required: bool


@dataclass(frozen=True)
class MonitoringConfig:
    weekly_wape_warning: float
    weekly_wape_severe: float
    reporting_timezone: str
    drift_can_trigger_retraining_alone: bool


@dataclass(frozen=True)
class ProductionConfig:
    version: int
    timezone: str
    expected_zones: int
    tracking: TrackingConfig
    storage: StorageConfig
    lifecycle: LifecycleConfig
    monitoring: MonitoringConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _tracking_uri(root: Path, value: str) -> str:
    prefix = "sqlite:///"
    if value.startswith(prefix):
        database = value[len(prefix) :]
        path = _resolve(root, database)
        return prefix + path.as_posix()
    return value


def load_production_config(
    path: str | Path,
    *,
    project_root: str | Path = ".",
) -> ProductionConfig:
    path = Path(path)
    root = Path(project_root).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required_sections = {"tracking", "storage", "lifecycle", "monitoring"}
    missing = required_sections.difference(payload)
    if missing:
        raise ValueError(f"production config is missing: {', '.join(sorted(missing))}")

    tracking = payload["tracking"]
    storage = payload["storage"]
    lifecycle = payload["lifecycle"]
    monitoring = payload["monitoring"]
    config = ProductionConfig(
        version=int(payload["version"]),
        timezone=str(payload["timezone"]),
        expected_zones=int(payload["expected_zones"]),
        tracking=TrackingConfig(
            uri=_tracking_uri(root, str(tracking["uri"])),
            artifact_root=_resolve(root, str(tracking["artifact_root"])),
            experiment_name=str(tracking["experiment_name"]),
            registered_model_name=str(tracking["registered_model_name"]),
            production_alias=str(tracking["production_alias"]),
        ),
        storage=StorageConfig(
            feature_root=_resolve(root, str(storage["feature_root"])),
            prediction_root=_resolve(root, str(storage["prediction_root"])),
            score_root=_resolve(root, str(storage["score_root"])),
            state_db=_resolve(root, str(storage["state_db"])),
        ),
        lifecycle=LifecycleConfig(
            strategy=str(lifecycle["strategy"]),
            training_history=str(lifecycle["training_history"]),
            retrain_every_calendar_days=int(lifecycle["retrain_every_calendar_days"]),
            statistical_promotion_gate=bool(lifecycle["statistical_promotion_gate"]),
            technical_validation_required=bool(lifecycle["technical_validation_required"]),
        ),
        monitoring=MonitoringConfig(
            weekly_wape_warning=float(monitoring["weekly_wape_warning"]),
            weekly_wape_severe=float(monitoring["weekly_wape_severe"]),
            reporting_timezone=str(monitoring["reporting_timezone"]),
            drift_can_trigger_retraining_alone=bool(
                monitoring["drift_can_trigger_retraining_alone"]
            ),
        ),
    )
    if config.version != 1:
        raise ValueError(f"unsupported production config version: {config.version}")
    if config.expected_zones < 1:
        raise ValueError("expected_zones must be positive")
    if config.lifecycle.retrain_every_calendar_days < 1:
        raise ValueError("retrain cadence must be positive")
    if config.monitoring.weekly_wape_severe <= config.monitoring.weekly_wape_warning:
        raise ValueError("severe WAPE threshold must exceed warning threshold")
    return config
