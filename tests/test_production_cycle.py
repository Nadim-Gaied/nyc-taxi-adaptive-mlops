from dataclasses import dataclass

import pandas as pd
import pytest

from taxi_mlops.production import (
    PredictionBatch,
    ProductionCycleRunner,
    ProductionStateStore,
    ScoreRecord,
    cycle_hours,
    load_production_config,
)


@dataclass(frozen=True)
class _Result:
    action: str

    def to_dict(self) -> dict[str, object]:
        return {"action": self.action}


class _Training:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, as_of):
        self.calls.append(f"train:{as_of.isoformat()}")
        return _Result("not_due")


class _Scoring:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, target, *, as_of):
        self.calls.append(f"score:{target.isoformat()}")
        return ScoreRecord(
            batch_id="previous",
            scored_at=as_of.isoformat(),
            scored_path="score.parquet",
            observations=2,
            actual_pickups=3.0,
            predicted_pickups=3.0,
            mae=0.0,
            wape=0.0,
            mean_bias=0.0,
            prediction_coverage=1.0,
            score_sha256="score-hash",
        )


class _Prediction:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, target):
        self.calls.append(f"predict:{target.isoformat()}")
        return _batch("current", target)


class _Monitoring:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def build(self, *, as_of):
        self.calls.append(f"monitor:{as_of.isoformat()}")
        return _MonitoringResult()


@dataclass(frozen=True)
class _MonitoringResult:
    forecast_status: str = "healthy"

    def to_dict(self) -> dict[str, object]:
        return {"forecast_status": self.forecast_status}


def _batch(batch_id: str, target: pd.Timestamp) -> PredictionBatch:
    return PredictionBatch(
        batch_id=batch_id,
        target_hour=target.isoformat(),
        generated_at=target.isoformat(),
        model_name="taxi-demand",
        model_version="1",
        model_alias="champion",
        model_run_id="run-1",
        model_uri="models:/taxi-demand/1",
        feature_fingerprint="features-1",
        prediction_path=f"{batch_id}.parquet",
        prediction_sha256="hash",
        row_count=2,
    )


def _config(tmp_path):
    path = tmp_path / "production.yaml"
    path.write_text(
        """
version: 1
timezone: America/New_York
expected_zones: 2
tracking:
  uri: sqlite:///mlflow.db
  artifact_root: artifacts
  experiment_name: production
  registered_model_name: taxi-demand
  reference_alias: accuracy-reference
  production_alias: champion
storage:
  feature_root: features
  prediction_root: predictions
  score_root: scores
  model_root: models
  state_db: state/production.db
lifecycle:
  strategy: fixed_periodic_28d
  training_history: recent_365d
  retrain_every_calendar_days: 28
  statistical_promotion_gate: false
  technical_validation_required: true
monitoring:
  weekly_wape_warning: 0.14
  weekly_wape_severe: 0.16
  reporting_timezone: America/New_York
  drift_can_trigger_retraining_alone: false
""".strip(),
        encoding="utf-8",
    )
    return load_production_config(path, project_root=tmp_path)


def test_cycle_orders_training_scoring_prediction_and_monitoring(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    prediction_target, scoring_target = cycle_hours("2025-01-01T06:00:00Z")
    state.record_prediction(_batch("previous", scoring_target))
    calls: list[str] = []
    runner = ProductionCycleRunner(
        config=config,
        state_store=state,
        training_pipeline=_Training(calls),
        scoring_pipeline=_Scoring(calls),
        prediction_pipeline=_Prediction(calls),
        monitoring_reporter=_Monitoring(calls),
    )

    result = runner.run(prediction_target)

    assert calls == [
        "train:2025-01-01T06:00:00+00:00",
        "score:2025-01-01T05:00:00+00:00",
        "predict:2025-01-01T06:00:00+00:00",
        "monitor:2025-01-01T06:00:00+00:00",
    ]
    assert result.training["action"] == "not_due"
    assert state.watermark_history()[-1]["stage"] == "cycle"
    assert state.event_history()[-1]["status"] == "completed"


def test_cycle_requires_prior_prediction_unless_bootstrapping(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    calls: list[str] = []
    runner = ProductionCycleRunner(
        config=config,
        state_store=state,
        training_pipeline=_Training(calls),
        scoring_pipeline=_Scoring(calls),
        prediction_pipeline=_Prediction(calls),
        monitoring_reporter=_Monitoring(calls),
    )

    with pytest.raises(ValueError, match="previous-hour prediction is missing"):
        runner.run("2025-01-01T06:00:00Z")
    assert state.event_history()[-1]["status"] == "failed"

    recovered = runner.run("2025-01-01T06:00:00Z", allow_missing_previous_prediction=True)
    assert recovered.scoring["action"] == "skipped_bootstrap"


def test_cycle_rejects_non_hour_boundary() -> None:
    with pytest.raises(ValueError, match="hour boundary"):
        cycle_hours("2025-01-01T06:30:00Z")
