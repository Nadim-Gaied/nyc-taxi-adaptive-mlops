"""Executable lifecycle strategies for the locked historical replay."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import pandas as pd

from taxi_mlops.evaluation import assign_demand_tiers, validate_common_targets
from taxi_mlops.models.training_history import _history_intervals
from taxi_mlops.replay.clock import shift_local_calendar_days
from taxi_mlops.replay.decisions import (
    FrozenPerformanceMonitor,
    PromotionReference,
    evaluate_promotion_candidate,
    prepare_promotion_reference,
)
from taxi_mlops.replay.engine import ReplayContext
from taxi_mlops.replay.metrics import ReplayMetricsAccumulator
from taxi_mlops.replay.modeling import ReplayModelStore


def _timestamp_id(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")


class ModelLifecycleStrategy:
    """Shared prediction, monitoring, training, and audit behavior."""

    name = "base"

    def __init__(
        self,
        *,
        model_store: ReplayModelStore,
        policy: dict[str, object],
        feature_columns: Sequence[str],
        feature_set: str,
        timezone: str,
        model_parameters: Mapping[str, object] | None = None,
        initial_history: str = "recent_365d",
    ) -> None:
        self.model_store = model_store
        self.policy = policy
        self.feature_columns = list(feature_columns)
        self.feature_set = feature_set
        self.timezone = timezone
        self.model_parameters = model_parameters
        self.initial_history = initial_history
        self.active_model_id: str | None = None
        self.monitor = FrozenPerformanceMonitor(policy)
        self._recent_scores: deque[pd.DataFrame] = deque()
        self._candidate_prediction_cache: dict[str, pd.DataFrame] = {}
        self.audit_rows: list[dict[str, object]] = []
        self.training_jobs = 0
        self.training_seconds = 0.0
        self.inference_seconds = 0.0
        self.registered_models_evaluated = 0
        self.candidate_evaluation_events = 0
        self.promotion_rejections = 0
        self.metrics: ReplayMetricsAccumulator | None = None

    def _train(
        self,
        context: ReplayContext,
        *,
        as_of: pd.Timestamp,
        history_strategy: str,
        purpose: str,
    ) -> str:
        intervals = _history_intervals(
            history_strategy,
            as_of,
            context.source.data_start,
            self.timezone,
        )
        history = context.source.observations_as_of(
            as_of, start=min(start for start, _ in intervals)
        )
        model_id = f"{self.name}__{purpose}__{history_strategy}__{_timestamp_id(as_of)}"
        outcome = self.model_store.train(
            history,
            model_id=model_id,
            as_of=as_of,
            registered_at=context.now,
            history_strategy=history_strategy,
            feature_set=self.feature_set,
            feature_columns=self.feature_columns,
            timezone=self.timezone,
            model_parameters=self.model_parameters,
        )
        self.training_jobs += 1
        self.training_seconds += outcome.training_seconds
        self.audit_rows.append(
            {
                "timestamp": context.now,
                "event": "model_trained",
                "model_id": model_id,
                "history_strategy": history_strategy,
                "purpose": purpose,
                "training_rows": outcome.training_rows,
                "training_seconds": outcome.training_seconds,
            }
        )
        return model_id

    def _promote(
        self,
        context: ReplayContext,
        model_id: str,
        *,
        reason: str,
        cooldown_weeks: int,
    ) -> bool:
        try:
            context.registry.promote(
                model_id,
                as_of=context.now,
                reason=reason,
                cooldown_weeks=cooldown_weeks,
            )
        except ValueError as error:
            if "cooldown" not in str(error):
                raise
            self.promotion_rejections += 1
            self.audit_rows.append(
                {
                    "timestamp": context.now,
                    "event": "promotion_rejected_cooldown",
                    "model_id": model_id,
                    "reason": reason,
                }
            )
            return False
        self.active_model_id = model_id
        self.audit_rows.append(
            {
                "timestamp": context.now,
                "event": "model_promoted",
                "model_id": model_id,
                "reason": reason,
            }
        )
        return True

    def initialize(self, context: ReplayContext) -> None:
        if self.model_store.registry is not context.registry:
            raise ValueError("model store and replay context must share one registry")
        tier_start = shift_local_calendar_days(context.now, days=-365, timezone=self.timezone)
        tier_reference = context.source.observations_as_of(context.now, start=tier_start)
        tiers = assign_demand_tiers(tier_reference)
        self.metrics = ReplayMetricsAccumulator(
            replay_start=context.now,
            timezone=self.timezone,
            zone_tiers=tiers,
        )
        model_id = self._train(
            context,
            as_of=context.now,
            history_strategy=self.initial_history,
            purpose="initial",
        )
        self._promote(context, model_id, reason="initial deployment", cooldown_weeks=0)

    def observe(self, context: ReplayContext, scored: pd.DataFrame) -> None:
        if self.metrics is None:
            raise RuntimeError("strategy metrics were not initialized")
        self.metrics.update(scored)
        self._recent_scores.append(scored.copy())
        keep_after = shift_local_calendar_days(context.now, days=-35, timezone=self.timezone)
        while self._recent_scores and self._recent_scores[0]["pickup_hour"].max() < keep_after:
            self._recent_scores.popleft()

    def predict(self, context: ReplayContext, features: pd.DataFrame) -> pd.DataFrame:
        if self.active_model_id is None:
            raise RuntimeError("strategy has no active model")
        prediction, seconds = self.model_store.predict(self.active_model_id, features)
        self.inference_seconds += seconds
        output = features[["pickup_hour", "pickup_zone_id"]].copy()
        output["prediction"] = prediction.to_numpy()
        output["model_id"] = self.active_model_id
        return output

    def _scored_window(self, context: ReplayContext, *, days: int) -> pd.DataFrame | None:
        start = shift_local_calendar_days(context.now, days=-days, timezone=self.timezone)
        frames = [
            frame.loc[frame["pickup_hour"].ge(start)]
            for frame in self._recent_scores
            if frame["pickup_hour"].max() >= start
        ]
        if not frames:
            return None
        scored = pd.concat(frames, ignore_index=True)
        expected = context.source.labels_between(start, context.now, as_of=context.now)
        try:
            validate_common_targets({"expected": expected, "scored": scored})
        except ValueError:
            return None
        return scored.sort_values(["pickup_hour", "pickup_zone_id"])

    def decide(self, context: ReplayContext) -> None:
        weekly = self._scored_window(context, days=7)
        if weekly is None:
            self.audit_rows.append(
                {"timestamp": context.now, "event": "monitor_deferred_incomplete_week"}
            )
            return
        decision = self.monitor.evaluate(weekly)
        self.audit_rows.append(
            {
                "timestamp": context.now,
                "event": "weekly_monitor",
                **decision.__dict__,
            }
        )
        self._handle_decision(context, decision.candidate_evaluation_trigger)

    def _handle_decision(self, context: ReplayContext, triggered: bool) -> None:
        del context, triggered

    def _candidate_predictions(
        self,
        context: ReplayContext,
        *,
        model_id: str,
        evaluation_start: pd.Timestamp,
    ) -> pd.DataFrame:
        target = context.source.observations_as_of(context.now, start=evaluation_start)
        keys = ["pickup_hour", "pickup_zone_id"]
        cached = self._candidate_prediction_cache.get(model_id)
        if cached is None:
            cached = pd.DataFrame(columns=[*keys, "prediction"])
        else:
            cached = cached.loc[cached["pickup_hour"].ge(evaluation_start)].copy()

        output = target[[*keys, "pickup_count"]].merge(
            cached,
            on=keys,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        missing = output["prediction"].isna()
        if missing.any():
            missing_keys = output.loc[missing, keys]
            missing_target = target.merge(
                missing_keys,
                on=keys,
                how="inner",
                sort=False,
                validate="one_to_one",
            )
            prediction, seconds = self.model_store.predict(model_id, missing_target)
            self.inference_seconds += seconds
            additions = missing_target[keys].copy()
            additions["prediction"] = prediction.to_numpy()
            cached = (
                additions.reset_index(drop=True)
                if cached.empty
                else pd.concat([cached, additions], ignore_index=True)
            )
            cached = cached.drop_duplicates(keys, keep="last")
            output = output.drop(columns="prediction").merge(
                cached,
                on=keys,
                how="left",
                sort=False,
                validate="one_to_one",
            )
        if output["prediction"].isna().any():
            raise ValueError("candidate prediction cache has incomplete coverage")
        self._candidate_prediction_cache[model_id] = cached
        return output.sort_values(keys).reset_index(drop=True)

    def _gate_model(
        self,
        context: ReplayContext,
        *,
        model_id: str,
        current: pd.DataFrame,
        evaluation_start: pd.Timestamp,
        source: str,
        promotion_reference: PromotionReference | None = None,
    ) -> dict[str, object]:
        candidate = self._candidate_predictions(
            context, model_id=model_id, evaluation_start=evaluation_start
        )
        if promotion_reference is None:
            promotion_reference, tier_reference = self._prepare_gate_reference(
                context, current=current, evaluation_start=evaluation_start
            )
        else:
            tier_reference = pd.DataFrame()
        gate = evaluate_promotion_candidate(
            current,
            candidate,
            tier_reference,
            self.policy,
            prepared_reference=promotion_reference,
        )
        self.candidate_evaluation_events += 1
        if source == "registry":
            self.registered_models_evaluated += 1
        self.audit_rows.append(
            {
                "timestamp": context.now,
                "event": "candidate_evaluated",
                "model_id": model_id,
                "candidate_source": source,
                **gate,
            }
        )
        return gate

    def _prepare_gate_reference(
        self,
        context: ReplayContext,
        *,
        current: pd.DataFrame,
        evaluation_start: pd.Timestamp,
    ) -> tuple[PromotionReference, pd.DataFrame]:
        tier_start = shift_local_calendar_days(evaluation_start, days=-365, timezone=self.timezone)
        tier_reference = context.source.observations_as_of(evaluation_start, start=tier_start)
        return (
            prepare_promotion_reference(current, tier_reference),
            tier_reference,
        )

    def _new_challenger(
        self,
        context: ReplayContext,
        current: pd.DataFrame,
        *,
        promotion_reference: PromotionReference | None = None,
    ) -> None:
        promotion = self.policy["promotion"]
        retraining = self.policy["retraining"]
        if not isinstance(promotion, dict) or not isinstance(retraining, dict):
            raise ValueError("policy is missing promotion or retraining settings")
        evaluation_start = shift_local_calendar_days(
            context.now,
            days=-int(promotion["evaluation_window_days"]),
            timezone=self.timezone,
        )
        evaluation_id = self._train(
            context,
            as_of=evaluation_start,
            history_strategy=str(retraining["challenger_history"]),
            purpose=f"evaluation_at_{_timestamp_id(context.now)}",
        )
        gate = self._gate_model(
            context,
            model_id=evaluation_id,
            current=current,
            evaluation_start=evaluation_start,
            source="new_challenger",
            promotion_reference=promotion_reference,
        )
        if not bool(gate["all_quality_gates_pass"]):
            self.promotion_rejections += 1
            return
        deployment_id = self._train(
            context,
            as_of=context.now,
            history_strategy=str(retraining["challenger_history"]),
            purpose="deployment",
        )
        self._promote(
            context,
            deployment_id,
            reason=f"challenger passed 28-day gate using {evaluation_id}",
            cooldown_weeks=int(promotion["cooldown_weeks"]),
        )

    def operational_summary(self) -> dict[str, object]:
        weekly = [row for row in self.audit_rows if row["event"] == "weekly_monitor"]
        degraded = [row for row in weekly if bool(row["warning_breach"])]
        return {
            "strategy": self.name,
            "training_jobs": self.training_jobs,
            "training_seconds": self.training_seconds,
            "inference_seconds": self.inference_seconds,
            "model_switches": max(0, len(self.model_store.registry.promotions()) - 1),
            "registered_models": len(self.model_store.registry.models()),
            "registered_models_evaluated": self.registered_models_evaluated,
            "candidate_evaluation_events": self.candidate_evaluation_events,
            "promotion_rejections": self.promotion_rejections,
            "degraded_weeks": len(degraded),
            "consecutive_degraded_weeks_max": max(
                (int(row["consecutive_warning_weeks"]) for row in weekly),
                default=0,
            ),
        }


class NeverRetrainStrategy(ModelLifecycleStrategy):
    name = "never_retrain"


class FixedPeriodicStrategy(ModelLifecycleStrategy):
    name = "fixed_periodic_28d"

    def __init__(self, *args, retrain_days: int = 28, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if retrain_days <= 0:
            raise ValueError("retrain_days must be positive")
        self.retrain_days = retrain_days
        self._next_retrain: pd.Timestamp | None = None

    def initialize(self, context: ReplayContext) -> None:
        super().initialize(context)
        self._next_retrain = shift_local_calendar_days(
            context.now, days=self.retrain_days, timezone=self.timezone
        )

    def _handle_decision(self, context: ReplayContext, triggered: bool) -> None:
        del triggered
        if self._next_retrain is None or context.now < self._next_retrain:
            return
        model_id = self._train(
            context,
            as_of=context.now,
            history_strategy="recent_365d",
            purpose="periodic",
        )
        self._promote(
            context,
            model_id,
            reason="fixed 28-day schedule",
            cooldown_weeks=0,
        )
        self._next_retrain = shift_local_calendar_days(
            self._next_retrain,
            days=self.retrain_days,
            timezone=self.timezone,
        )


class PerformanceTriggeredStrategy(ModelLifecycleStrategy):
    name = "performance_triggered"

    def _handle_decision(self, context: ReplayContext, triggered: bool) -> None:
        if not triggered:
            return
        current = self._scored_window(
            context, days=int(self.policy["promotion"]["evaluation_window_days"])
        )
        if current is None:
            self.audit_rows.append(
                {"timestamp": context.now, "event": "trigger_deferred_incomplete_gate_window"}
            )
            return
        self._new_challenger(context, current)


class AdaptiveReuseStrategy(ModelLifecycleStrategy):
    name = "adaptive_reuse_and_retrain"

    def _handle_decision(self, context: ReplayContext, triggered: bool) -> None:
        if not triggered:
            return
        promotion = self.policy["promotion"]
        if not isinstance(promotion, dict):
            raise ValueError("policy is missing promotion settings")
        current = self._scored_window(context, days=int(promotion["evaluation_window_days"]))
        if current is None:
            self.audit_rows.append(
                {"timestamp": context.now, "event": "trigger_deferred_incomplete_gate_window"}
            )
            return
        evaluation_start = shift_local_calendar_days(
            context.now,
            days=-int(promotion["evaluation_window_days"]),
            timezone=self.timezone,
        )
        promotion_reference, _ = self._prepare_gate_reference(
            context,
            current=current,
            evaluation_start=evaluation_start,
        )
        passing: list[tuple[float, str]] = []
        for record in context.registry.eligible_for_evaluation(
            evaluation_start=evaluation_start, as_of=context.now
        ):
            if record.model_id == self.active_model_id:
                continue
            gate = self._gate_model(
                context,
                model_id=record.model_id,
                current=current,
                evaluation_start=evaluation_start,
                source="registry",
                promotion_reference=promotion_reference,
            )
            if bool(gate["all_quality_gates_pass"]):
                passing.append((float(gate["relative_wape_improvement"]), record.model_id))
        if passing:
            _, best_model = max(passing)
            self._promote(
                context,
                best_model,
                reason="best eligible registered model passed 28-day gate",
                cooldown_weeks=int(promotion["cooldown_weeks"]),
            )
            return
        self._new_challenger(
            context,
            current,
            promotion_reference=promotion_reference,
        )
