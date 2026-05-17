"""Prefect deployments for hourly production and daily integrity checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from prefect import flow, task

from taxi_mlops.production import (
    ProductionCycleRunner,
    ProductionIntegrityAuditor,
    ProductionStateStore,
    create_control_plane_checkpoint,
    load_production_config,
)


def _hour_boundary(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").floor("h")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("as_of must include an explicit UTC offset")
    return timestamp.tz_convert("UTC")


@task(retries=2, retry_delay_seconds=[15, 60])
def run_cycle_task(
    config_path: str,
    project_root: str,
    as_of: str,
    allow_missing_previous_prediction: bool,
) -> dict[str, object]:
    config = load_production_config(config_path, project_root=project_root)
    state = ProductionStateStore(config.storage.state_db)
    runner = ProductionCycleRunner(config=config, state_store=state)
    return runner.run(
        as_of,
        allow_missing_previous_prediction=allow_missing_previous_prediction,
    ).to_dict()


@flow(name="nyc-taxi-hourly-production", log_prints=True)
def hourly_production_flow(
    config_path: str = "configs/production.yaml",
    project_root: str = ".",
    as_of: str | None = None,
    allow_missing_previous_prediction: bool = False,
) -> dict[str, object]:
    """Run the due training check, delayed score, prediction, and monitoring."""

    cutoff = _hour_boundary(as_of)
    return run_cycle_task(
        config_path,
        project_root,
        cutoff.isoformat(),
        allow_missing_previous_prediction,
    )


@task(retries=1, retry_delay_seconds=60)
def audit_and_checkpoint_task(
    config_path: str,
    project_root: str,
    checkpoint_root: str,
) -> dict[str, object]:
    config = load_production_config(config_path, project_root=project_root)
    state = ProductionStateStore(config.storage.state_db)
    audit = ProductionIntegrityAuditor(config=config, state_store=state).run()
    if audit.status == "failed":
        raise RuntimeError("production integrity audit failed")
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    checkpoint = create_control_plane_checkpoint(
        config=config,
        state_store=state,
        destination=Path(project_root) / checkpoint_root / stamp,
    )
    return {"audit": audit.to_dict(), "checkpoint": checkpoint.to_dict()}


@flow(name="nyc-taxi-daily-recovery-checkpoint", log_prints=True)
def daily_recovery_flow(
    config_path: str = "configs/production.yaml",
    project_root: str = ".",
    checkpoint_root: str = "data/production/checkpoints",
) -> dict[str, object]:
    """Audit the production state and create a restorable control-plane copy."""

    return audit_and_checkpoint_task(config_path, project_root, checkpoint_root)
