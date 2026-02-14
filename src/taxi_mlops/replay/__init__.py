"""Leakage-safe historical production replay primitives."""

from taxi_mlops.replay.clock import ReplayClock, shift_local_calendar_days
from taxi_mlops.replay.decisions import (
    FrozenPerformanceMonitor,
    MonitorDecision,
    evaluate_promotion_candidate,
)
from taxi_mlops.replay.engine import ReplayContext, ReplayEngine, ReplayStrategy
from taxi_mlops.replay.features import build_monthly_feature_store
from taxi_mlops.replay.metrics import ReplayMetricsAccumulator
from taxi_mlops.replay.modeling import ReplayModelStore, TrainingOutcome
from taxi_mlops.replay.monthly_source import MonthlyReplayDataSource
from taxi_mlops.replay.registry import ModelRecord, ModelRegistry, PromotionRecord
from taxi_mlops.replay.reporting import write_locked_replay_report
from taxi_mlops.replay.runner import run_locked_replay
from taxi_mlops.replay.seeding import seed_replay_registry
from taxi_mlops.replay.source import PredictionLedger, ReplayDataSource
from taxi_mlops.replay.strategies import (
    AdaptiveReuseStrategy,
    FixedPeriodicStrategy,
    ModelLifecycleStrategy,
    NeverRetrainStrategy,
    PerformanceTriggeredStrategy,
)

__all__ = [
    "ModelRecord",
    "ModelRegistry",
    "ModelLifecycleStrategy",
    "NeverRetrainStrategy",
    "FrozenPerformanceMonitor",
    "MonitorDecision",
    "MonthlyReplayDataSource",
    "PredictionLedger",
    "PromotionRecord",
    "ReplayClock",
    "ReplayContext",
    "ReplayDataSource",
    "ReplayEngine",
    "ReplayModelStore",
    "ReplayMetricsAccumulator",
    "ReplayStrategy",
    "TrainingOutcome",
    "build_monthly_feature_store",
    "AdaptiveReuseStrategy",
    "FixedPeriodicStrategy",
    "PerformanceTriggeredStrategy",
    "evaluate_promotion_candidate",
    "seed_replay_registry",
    "run_locked_replay",
    "shift_local_calendar_days",
    "write_locked_replay_report",
]
