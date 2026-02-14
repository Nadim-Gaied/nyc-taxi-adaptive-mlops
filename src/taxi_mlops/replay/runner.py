"""Orchestrate the frozen four-strategy locked replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from taxi_mlops.replay.clock import ReplayClock
from taxi_mlops.replay.engine import ReplayEngine
from taxi_mlops.replay.modeling import ReplayModelStore
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource
from taxi_mlops.replay.registry import ModelRegistry
from taxi_mlops.replay.source import PredictionLedger
from taxi_mlops.replay.strategies import (
    AdaptiveReuseStrategy,
    FixedPeriodicStrategy,
    NeverRetrainStrategy,
    PerformanceTriggeredStrategy,
)

STRATEGY_CLASSES = {
    "never_retrain": NeverRetrainStrategy,
    "fixed_periodic_28d": FixedPeriodicStrategy,
    "performance_triggered": PerformanceTriggeredStrategy,
    "adaptive_reuse_and_retrain": AdaptiveReuseStrategy,
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def run_locked_replay(
    *,
    feature_root: str | Path,
    output_dir: str | Path,
    seed_registry_path: str | Path,
    plan: dict[str, object],
    policy: dict[str, object],
    history_start: str | pd.Timestamp,
    feature_columns: Sequence[str],
    feature_set: str,
    model_parameters: Mapping[str, object] | None,
    strategy_names: Sequence[str] | None = None,
    resume_completed: bool = False,
) -> pd.DataFrame:
    """Run frozen strategies sequentially and write bounded result artifacts."""

    if plan.get("status") != "frozen_ready_for_locked_replay":
        raise ValueError("locked replay plan is not frozen and ready")
    clock_config = plan["clock"]
    initialization = plan["initialization"]
    if not isinstance(clock_config, dict) or not isinstance(initialization, dict):
        raise ValueError("replay plan is missing clock or initialization")
    selected = list(strategy_names or STRATEGY_CLASSES)
    unknown = sorted(set(selected).difference(STRATEGY_CLASSES))
    if unknown:
        raise ValueError(f"unknown replay strategies: {', '.join(unknown)}")
    if set(selected) != set(STRATEGY_CLASSES):
        raise ValueError("locked replay must execute all four frozen strategies")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for strategy_name in selected:
        strategy_dir = output_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        summary_path = strategy_dir / "summary.json"
        required_outputs = [
            strategy_dir / "events.csv",
            strategy_dir / "lifecycle_audit.csv",
            strategy_dir / "weekly_metrics.csv",
            strategy_dir / "tier_metrics.csv",
            strategy_dir / "zone_metrics.csv",
            strategy_dir / "registry.json",
            summary_path,
        ]
        if resume_completed and all(path.is_file() for path in required_outputs):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("strategy") != strategy_name:
                raise ValueError(
                    f"completed summary does not match strategy: {strategy_name}"
                )
            if int(summary.get("observations", 0)) <= 0:
                raise ValueError(
                    f"completed strategy has no scored observations: {strategy_name}"
                )
            summaries.append(summary)
            print(f"Resumed completed locked replay strategy: {strategy_name}", flush=True)
            continue
        if strategy_name == "adaptive_reuse_and_retrain":
            registry = ModelRegistry.from_manifest(seed_registry_path)
        else:
            registry = ModelRegistry()
        source = MonthlyReplayDataSource(
            feature_root=feature_root,
            start=history_start,
            end=clock_config["end"],
            timezone=str(plan["timezone"]),
        )
        store = ReplayModelStore(strategy_dir / "models", registry)
        strategy_class = STRATEGY_CLASSES[strategy_name]
        strategy = strategy_class(
            model_store=store,
            policy=policy,
            feature_columns=feature_columns,
            feature_set=feature_set,
            timezone=str(plan["timezone"]),
            model_parameters=model_parameters,
            initial_history=str(initialization["active_model_history"]),
        )
        engine = ReplayEngine(
            clock=ReplayClock(
                start=clock_config["start"],
                end=clock_config["end"],
                timezone=str(plan["timezone"]),
                decision_cadence_days=int(clock_config["decision_cadence_days"]),
            ),
            source=source,
            strategy=strategy,
            registry=registry,
            ledger=PredictionLedger(retain_scored=False),
        )
        events = engine.run()
        if strategy.metrics is None:
            raise RuntimeError("strategy completed without metrics")
        overall = strategy.metrics.overall_metrics()
        weekly = strategy.metrics.weekly_metrics()
        tier = strategy.metrics.tier_metrics()
        zone = strategy.metrics.zone_metrics()
        operational = strategy.operational_summary()
        audit = pd.DataFrame(strategy.audit_rows)
        events.to_csv(strategy_dir / "events.csv", index=False)
        audit.to_csv(strategy_dir / "lifecycle_audit.csv", index=False)
        weekly.to_csv(strategy_dir / "weekly_metrics.csv", index=False)
        tier.to_csv(strategy_dir / "tier_metrics.csv", index=False)
        zone.to_csv(strategy_dir / "zone_metrics.csv", index=False)
        registry.write_manifest(strategy_dir / "registry.json")
        summary = {
            **operational,
            **overall,
            "weekly_wape_p90": float(weekly["wape"].quantile(0.90)),
            "worst_week_wape": float(weekly["wape"].max()),
        }
        _write_json(summary_path, summary)
        summaries.append(summary)
        print(f"Completed locked replay strategy: {strategy_name}", flush=True)

    comparison = pd.DataFrame(summaries).sort_values("wape").reset_index(drop=True)
    if comparison["observations"].nunique() != 1:
        raise ValueError("replay strategies did not score identical observation counts")
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
    _write_json(
        output_dir / "summary.json",
        {
            "strategies": selected,
            "observations_per_strategy": int(comparison.iloc[0]["observations"]),
            "best_overall_wape_strategy": str(comparison.iloc[0]["strategy"]),
            "locked_protocol_changed": False,
        },
    )
    return comparison
