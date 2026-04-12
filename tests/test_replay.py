from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from taxi_mlops.features import build_feature_table
from taxi_mlops.policies import load_lifecycle_policy, load_locked_replay_plan
from taxi_mlops.replay import (
    AdaptiveReuseStrategy,
    FixedPeriodicStrategy,
    FrozenPerformanceMonitor,
    ModelRecord,
    ModelRegistry,
    MonthlyReplayDataSource,
    PerformanceTriggeredStrategy,
    PredictionLedger,
    ReplayClock,
    ReplayDataSource,
    ReplayEngine,
    ReplayModelStore,
    TrainingOutcome,
    build_monthly_feature_store,
    evaluate_promotion_candidate,
    run_locked_replay,
    shift_local_calendar_days,
)


def _feature_table() -> pd.DataFrame:
    hours = pd.date_range(
        "2020-01-01T00:00:00Z",
        "2020-01-03T02:00:00Z",
        freq="h",
        inclusive="left",
    )
    return pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [
                zone * 10 + index % 24 for zone in (1, 2) for index in range(len(hours))
            ],
            "lag_1h": [0.0] * (2 * len(hours)),
        }
    )


def test_replay_source_hides_current_and_future_labels() -> None:
    source = ReplayDataSource(_feature_table())
    now = pd.Timestamp("2020-01-02T00:00:00Z")

    history = source.observations_as_of(now)
    features = source.prediction_features(now, as_of=now)

    assert history["pickup_hour"].max() < now
    assert "pickup_count" not in features.columns
    assert features["pickup_hour"].eq(now).all()
    with pytest.raises(ValueError, match="current clock hour"):
        source.prediction_features(now + pd.Timedelta(hours=1), as_of=now)
    with pytest.raises(ValueError, match="not yet released"):
        source.labels_between(now, now + pd.Timedelta(hours=1), as_of=now)


def test_prediction_ledger_enforces_generation_time_and_scores_later() -> None:
    source = ReplayDataSource(_feature_table())
    ledger = PredictionLedger()
    target = pd.Timestamp("2020-01-02T00:00:00Z")
    predictions = pd.DataFrame(
        {
            "pickup_hour": [target, target],
            "pickup_zone_id": [1, 2],
            "prediction": [10.0, 20.0],
        }
    )

    ledger.record(predictions, generated_at=target, model_id="model-1")
    assert ledger.score_available(source, as_of=target).empty
    scored = ledger.score_available(source, as_of=target + pd.Timedelta(hours=1))
    assert len(scored) == 2
    assert scored["pickup_count"].notna().all()
    with pytest.raises(ValueError, match="after their target"):
        PredictionLedger().record(
            predictions,
            generated_at=target + pd.Timedelta(hours=1),
            model_id="late",
        )


def test_registry_excludes_models_trained_inside_evaluation_window_and_cools_down(
    tmp_path,
) -> None:
    registry = ModelRegistry()
    for model_id, trained_at in (
        ("old", "2020-01-01T00:00:00Z"),
        ("too-new", "2020-01-15T00:00:00Z"),
    ):
        registry.register(
            ModelRecord(
                model_id=model_id,
                trained_at=trained_at,
                training_start="2019-01-01T00:00:00Z",
                training_end=trained_at,
                history_strategy="recent_365d",
                feature_set="full_v1",
                artifact_path=f"models/{model_id}.txt",
            )
        )

    eligible = registry.eligible_for_evaluation(
        evaluation_start="2020-01-10T00:00:00Z",
        as_of="2020-02-01T00:00:00Z",
    )
    assert [record.model_id for record in eligible] == ["old"]
    registry.promote(
        "old",
        as_of="2020-02-01T00:00:00Z",
        reason="initial",
        cooldown_weeks=0,
    )
    with pytest.raises(ValueError, match="cooldown"):
        registry.promote(
            "too-new",
            as_of="2020-02-05T00:00:00Z",
            reason="too soon",
            cooldown_weeks=1,
        )
    registry.promote(
        "too-new",
        as_of="2020-02-08T00:00:00Z",
        reason="eligible",
        cooldown_weeks=1,
    )
    assert registry.active_model_id == "too-new"
    manifest = tmp_path / "registry.json"
    registry.write_manifest(manifest)
    restored = ModelRegistry.from_manifest(manifest)
    assert restored.active_model_id == "too-new"
    assert restored.models() == registry.models()
    assert restored.promotions() == registry.promotions()


def test_replay_engine_orders_release_decision_and_prediction_without_future_access() -> None:
    class RecordingStrategy:
        name = "recording"

        def __init__(self) -> None:
            self.observed_rows = 0
            self.decisions = 0

        def initialize(self, context) -> None:
            assert context.history()["pickup_hour"].max() < context.now

        def observe(self, context, scored) -> None:
            assert scored["pickup_hour"].max() < context.now
            self.observed_rows += len(scored)

        def decide(self, context) -> None:
            assert context.ledger.frame()["scored"].any()
            self.decisions += 1

        def predict(self, context, features) -> pd.DataFrame:
            assert "pickup_count" not in features
            output = features[["pickup_hour", "pickup_zone_id"]].copy()
            output["prediction"] = 0.0
            output["model_id"] = "synthetic-model"
            return output

    strategy = RecordingStrategy()
    engine = ReplayEngine(
        clock=ReplayClock(
            start="2020-01-02T00:00:00Z",
            end="2020-01-03T02:00:00Z",
            timezone="UTC",
            decision_cadence_days=1,
        ),
        source=ReplayDataSource(_feature_table()),
        strategy=strategy,
    )
    events = engine.run()

    assert strategy.observed_rows == 52
    assert strategy.decisions == 1
    assert engine.ledger.frame()["scored"].all()
    decision_time = events.loc[events["event"].eq("lifecycle_decision"), "timestamp"].iloc[0]
    at_decision = events.loc[events["timestamp"].eq(decision_time), "event"].tolist()
    assert at_decision == [
        "labels_scored",
        "lifecycle_decision",
        "predictions_generated",
    ]


def test_clock_uses_local_calendar_days_across_spring_dst() -> None:
    clock = ReplayClock(
        start="2020-03-08T00:00:00-05:00",
        end="2020-03-10T00:00:00-04:00",
        decision_cadence_days=1,
    )
    for _ in range(23):
        clock.advance()

    assert clock.decision_due()
    assert str(clock.now) == "2020-03-09 04:00:00+00:00"


def test_calendar_day_shift_resolves_dst_gaps_and_folds_by_project_policy() -> None:
    spring = shift_local_calendar_days(
        "2020-04-12T06:00:00Z",
        days=-35,
        timezone="America/New_York",
    )
    autumn = shift_local_calendar_days(
        "2020-11-08T06:00:00Z",
        days=-7,
        timezone="America/New_York",
    )

    assert spring == pd.Timestamp("2020-03-08T07:00:00Z")
    assert autumn == pd.Timestamp("2020-11-01T06:00:00Z")


def test_locked_replay_plan_freezes_strategies_and_keeps_2020(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "configs/locked_replay_experiment_v1.yaml"
    plan = load_locked_replay_plan(source)

    assert set(plan["strategies"]) == {
        "never_retrain",
        "fixed_periodic_28d",
        "performance_triggered",
        "adaptive_reuse_and_retrain",
    }
    assert plan["clock"]["start"].startswith("2020-01-01")
    assert plan["access"]["open_locked_replay_during_engine_development"] is False

    invalid_plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    invalid_plan["interpretation_contract"]["tune_policy_during_locked_replay"] = True
    invalid = tmp_path / "locked_replay.yaml"
    invalid.write_text(yaml.safe_dump(invalid_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="tuning"):
        load_locked_replay_plan(invalid)


def test_frozen_monitor_and_promotion_gate_use_common_holdout() -> None:
    policy = {
        "performance_monitor": {
            "weekly_wape_warning": 0.10,
            "weekly_wape_severe": 0.20,
            "persistence_weeks": 2,
        },
        "promotion": {
            "required_prediction_coverage": 1.0,
            "minimum_relative_wape_improvement": 0.01,
            "maximum_tier_wape_regression": 0.005,
            "maximum_zone_mae_p95_regression": 1.0,
            "require_bias_not_worse": True,
        },
        "demand_tier_monitor": {},
    }
    weekly = pd.DataFrame(
        {
            "pickup_zone_id": [1, 2],
            "pickup_count": [10.0, 20.0],
            "prediction": [12.0, 22.0],
        }
    )
    monitor = FrozenPerformanceMonitor(policy)
    assert not monitor.evaluate(weekly).candidate_evaluation_trigger
    assert monitor.evaluate(weekly).candidate_evaluation_trigger

    target_hours = pd.to_datetime(["2020-01-01T00:00:00Z"] * 2)
    current = pd.DataFrame(
        {
            "pickup_hour": target_hours,
            "pickup_zone_id": [1, 2],
            "pickup_count": [100.0, 50.0],
            "prediction": [110.0, 55.0],
        }
    )
    candidate = current.copy()
    candidate["prediction"] = [102.0, 51.0]
    tier_reference = pd.DataFrame({"pickup_zone_id": [1, 2], "pickup_count": [1000.0, 100.0]})

    gate = evaluate_promotion_candidate(current, candidate, tier_reference, policy)
    assert gate["all_quality_gates_pass"]
    assert gate["relative_wape_improvement"] > 0.01


def test_fixed_periodic_strategy_trains_persists_and_switches(tmp_path) -> None:
    hours = pd.date_range(
        "2020-01-01T00:00:00Z",
        "2020-05-09T00:00:00Z",
        freq="h",
        inclusive="left",
    )
    canonical = pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [
                zone * 5 + index % 24 + index % 7 for zone in (1, 2) for index in range(len(hours))
            ],
        }
    )
    features = build_feature_table(canonical, timezone="UTC")
    registry = ModelRegistry()
    store = ReplayModelStore(tmp_path / "models", registry)
    project_root = Path(__file__).resolve().parents[1]
    policy = load_lifecycle_policy(project_root / "configs/lifecycle_policy_final_v1.yaml")
    strategy = FixedPeriodicStrategy(
        model_store=store,
        policy=policy,
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h", "lag_168h"],
        feature_set="synthetic",
        timezone="UTC",
        model_parameters={"n_estimators": 5, "n_jobs": 1},
        initial_history="recent_90d",
        retrain_days=7,
    )
    engine = ReplayEngine(
        clock=ReplayClock(
            start="2020-05-01T00:00:00Z",
            end="2020-05-09T00:00:00Z",
            timezone="UTC",
            decision_cadence_days=7,
        ),
        source=ReplayDataSource(features),
        strategy=strategy,
        registry=registry,
    )
    engine.run()

    summary = strategy.operational_summary()
    assert summary["training_jobs"] == 2
    assert summary["model_switches"] == 1
    assert summary["registered_models"] == 2
    assert len(list((tmp_path / "models").glob("*.txt"))) == 2
    assert engine.ledger.frame()["scored"].all()


class _FakeModelStore:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def train(
        self,
        feature_table,
        *,
        model_id,
        as_of,
        registered_at,
        history_strategy,
        feature_set,
        feature_columns,
        timezone,
        model_parameters,
    ) -> TrainingOutcome:
        del feature_columns, timezone, model_parameters
        cutoff = pd.Timestamp(as_of).tz_convert("UTC")
        registration_time = pd.Timestamp(registered_at).tz_convert("UTC")
        record = ModelRecord(
            model_id=model_id,
            trained_at=str(registration_time),
            training_start=str(feature_table["pickup_hour"].min()),
            training_end=str(cutoff),
            history_strategy=history_strategy,
            feature_set=feature_set,
            artifact_path=f"models/{model_id}.txt",
        )
        self.registry.register(record)
        return TrainingOutcome(record, len(feature_table), 0.01)

    def predict(self, model_id, features):
        value = 0.0 if "initial" in model_id else 10.0
        return pd.Series(value, index=features.index), 0.001


def _constant_replay_features() -> pd.DataFrame:
    hours = pd.date_range(
        "2019-01-01T00:00:00Z",
        "2020-02-01T00:00:00Z",
        freq="h",
        inclusive="left",
    )
    return pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [10.0] * (2 * len(hours)),
            "hour_of_day": list(hours.hour) * 2,
            "lag_1h": [10.0] * (2 * len(hours)),
        }
    )


def _synthetic_final_policy() -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    return load_lifecycle_policy(project_root / "configs/lifecycle_policy_final_v1.yaml")


def test_performance_triggered_strategy_uses_holdout_then_refits() -> None:
    registry = ModelRegistry()
    store = _FakeModelStore(registry)
    strategy = PerformanceTriggeredStrategy(
        model_store=store,
        policy=_synthetic_final_policy(),
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h"],
        feature_set="synthetic",
        timezone="UTC",
        initial_history="recent_90d",
    )
    engine = ReplayEngine(
        clock=ReplayClock(
            start="2020-01-01T00:00:00Z",
            end="2020-01-30T00:00:00Z",
            timezone="UTC",
            decision_cadence_days=7,
        ),
        source=ReplayDataSource(_constant_replay_features()),
        strategy=strategy,
        registry=registry,
    )
    engine.run()

    summary = strategy.operational_summary()
    assert summary["training_jobs"] == 3
    assert summary["model_switches"] == 1
    trained = [row for row in strategy.audit_rows if row["event"] == "model_trained"]
    evaluation = next(row for row in trained if "evaluation_at" in row["purpose"])
    deployment = next(row for row in trained if row["purpose"] == "deployment")
    evaluation_record = registry.records()[1]
    deployment_record = registry.records()[2]
    assert pd.Timestamp(evaluation["timestamp"]) == pd.Timestamp(evaluation_record.trained_at)
    assert pd.Timestamp(evaluation_record.trained_at) - pd.Timestamp(
        evaluation_record.training_end
    ) == pd.Timedelta(days=28)
    assert pd.Timestamp(deployment["timestamp"]) == pd.Timestamp(deployment_record.trained_at)


def test_adaptive_strategy_reuses_eligible_old_model_before_retraining() -> None:
    registry = ModelRegistry()
    seed = ModelRecord(
        model_id="seed-good",
        trained_at="2019-12-01T00:00:00Z",
        training_start="2019-01-01T00:00:00Z",
        training_end="2019-12-01T00:00:00Z",
        history_strategy="seasonal_90d_1y_gap",
        feature_set="synthetic",
        artifact_path="models/seed-good.txt",
    )
    registry.register(seed)
    store = _FakeModelStore(registry)
    strategy = AdaptiveReuseStrategy(
        model_store=store,
        policy=_synthetic_final_policy(),
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h"],
        feature_set="synthetic",
        timezone="UTC",
        initial_history="recent_90d",
    )
    engine = ReplayEngine(
        clock=ReplayClock(
            start="2020-01-01T00:00:00Z",
            end="2020-01-30T00:00:00Z",
            timezone="UTC",
            decision_cadence_days=7,
        ),
        source=ReplayDataSource(_constant_replay_features()),
        strategy=strategy,
        registry=registry,
    )
    engine.run()

    summary = strategy.operational_summary()
    assert registry.active_model_id == "seed-good"
    assert summary["training_jobs"] == 1
    assert summary["model_switches"] == 1
    assert summary["registered_models_evaluated"] >= 1


def test_monthly_feature_store_preserves_boundary_history_and_future_gate(
    tmp_path,
) -> None:
    canonical_root = tmp_path / "canonical"
    feature_root = tmp_path / "features"
    for month, start, end in (
        ("2020-01", "2020-01-30T00:00:00Z", "2020-02-01T00:00:00Z"),
        ("2020-02", "2020-02-01T00:00:00Z", "2020-02-03T00:00:00Z"),
    ):
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        frame = pd.DataFrame(
            {
                "pickup_hour": list(hours) * 2,
                "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
                "pickup_count": [
                    zone * 100 + index for zone in (1, 2) for index in range(len(hours))
                ],
            }
        )
        path = canonical_root / "2020" / f"zone_hour_{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    report = build_monthly_feature_store(
        canonical_root=canonical_root,
        feature_root=feature_root,
        start_month="2020-01",
        end_month="2020-02",
        timezone="UTC",
    )
    assert report["status"].eq("written").all()
    february = pd.read_parquet(feature_root / "2020" / "features_2020-02.parquet")
    first_hour = february.loc[
        february["pickup_hour"].eq(pd.Timestamp("2020-02-01T00:00:00Z"))
    ].sort_values("pickup_zone_id")
    assert first_hour["lag_1h"].tolist() == [147.0, 247.0]

    source = MonthlyReplayDataSource(
        feature_root=feature_root,
        start="2020-01-30T00:00:00Z",
        end="2020-02-03T00:00:00Z",
        timezone="UTC",
    )
    now = pd.Timestamp("2020-02-01T00:00:00Z")
    assert source.observations_as_of(now)["pickup_hour"].max() < now
    prediction_features = source.prediction_features(now, as_of=now)
    assert "pickup_count" not in prediction_features
    assert prediction_features["lag_1h"].tolist() == [147.0, 247.0]


def test_locked_replay_runner_compares_all_strategies_on_identical_targets(
    tmp_path,
) -> None:
    hours = pd.date_range(
        "2019-01-01T00:00:00Z",
        "2020-01-03T00:00:00Z",
        freq="h",
        inclusive="left",
    )
    canonical = pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [
                zone * 5 + index % 24 + index % 7 for zone in (1, 2) for index in range(len(hours))
            ],
        }
    )
    features = build_feature_table(canonical, timezone="UTC")
    feature_root = tmp_path / "features"
    for month, frame in features.groupby(features["pickup_hour"].dt.strftime("%Y-%m"), sort=True):
        path = feature_root / month[:4] / f"features_{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    seed_registry = ModelRegistry()
    seed_manifest = tmp_path / "seed_registry.json"
    seed_registry.write_manifest(seed_manifest)
    plan = {
        "status": "frozen_ready_for_locked_replay",
        "timezone": "UTC",
        "clock": {
            "start": "2020-01-01T00:00:00Z",
            "end": "2020-01-03T00:00:00Z",
            "decision_cadence_days": 7,
        },
        "initialization": {"active_model_history": "recent_365d"},
    }
    comparison = run_locked_replay(
        feature_root=feature_root,
        output_dir=tmp_path / "output",
        seed_registry_path=seed_manifest,
        plan=plan,
        policy=_synthetic_final_policy(),
        history_start="2019-01-01T00:00:00Z",
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h", "lag_168h"],
        feature_set="synthetic",
        model_parameters={"n_estimators": 3, "n_jobs": 1},
    )

    assert set(comparison["strategy"]) == {
        "never_retrain",
        "fixed_periodic_28d",
        "performance_triggered",
        "adaptive_reuse_and_retrain",
    }
    assert comparison["observations"].nunique() == 1
    assert comparison.iloc[0]["observations"] == 96
    assert (tmp_path / "output" / "strategy_comparison.csv").is_file()

    resumed = run_locked_replay(
        feature_root=tmp_path / "missing-features",
        output_dir=tmp_path / "output",
        seed_registry_path=seed_manifest,
        plan=plan,
        policy=_synthetic_final_policy(),
        history_start="2019-01-01T00:00:00Z",
        feature_columns=["pickup_zone_id", "hour_of_day", "lag_1h", "lag_168h"],
        feature_set="synthetic",
        model_parameters={"n_estimators": 3, "n_jobs": 1},
        resume_completed=True,
    )
    assert resumed["observations"].tolist() == comparison["observations"].tolist()
