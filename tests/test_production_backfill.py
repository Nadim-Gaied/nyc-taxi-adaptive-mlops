import numpy as np
import pandas as pd
import pytest

from taxi_mlops.production import (
    ModelDeployment,
    ProductionBackfillRunner,
    ProductionStateStore,
    ResolvedModel,
    load_production_config,
)


class _FakeBooster:
    pandas_categorical = [[1, 2]]

    def feature_name(self) -> list[str]:
        return ["pickup_zone_id", "lag_1h"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["lag_1h"].to_numpy(dtype=float) + 1.0


class _FakeResolver:
    def resolve(self) -> ResolvedModel:
        return ResolvedModel(
            name="taxi-demand",
            version="1",
            alias="champion",
            run_id="run-1",
            source_uri="models:/taxi-demand/1",
            booster=_FakeBooster(),
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
  monitoring_root: monitoring
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
  alert_window_hours: 3
  zone_lookback_days: 28
""".strip(),
        encoding="utf-8",
    )
    return load_production_config(path, project_root=tmp_path)


def _write_features(root, start: pd.Timestamp, hours: int) -> None:
    targets = pd.date_range(start, periods=hours, freq="h")
    rows = []
    for target in targets:
        rows.extend(
            [
                {
                    "pickup_hour": target,
                    "pickup_zone_id": 1,
                    "pickup_count": 11.0,
                    "lag_1h": 10.0,
                },
                {
                    "pickup_hour": target,
                    "pickup_zone_id": 2,
                    "pickup_count": 21.0,
                    "lag_1h": 20.0,
                },
            ]
        )
    frame = pd.DataFrame(rows)
    month = start.strftime("%Y-%m")
    path = root / month[:4] / f"features_{month}.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)


def test_backfill_is_bounded_complete_and_resumable(tmp_path) -> None:
    config = _config(tmp_path)
    start = pd.Timestamp("2025-01-01T05:00:00Z")
    end = start + pd.Timedelta(hours=3)
    _write_features(config.storage.feature_root, start, 3)
    state = ProductionStateStore(config.storage.state_db)
    runner = ProductionBackfillRunner(
        config=config,
        state_store=state,
        model_resolver=_FakeResolver(),
    )

    first = runner.run(start=start, end=end)
    second = runner.run(start=start, end=end)

    assert first.hours == 3
    assert first.predictions_created == 3
    assert first.scores_created == 3
    assert first.monitoring["alert_window_complete"] is True
    assert first.monitoring["forecast_status"] == "healthy"
    assert first.monitoring["operational_status"] == "healthy"
    assert second.predictions_created == 0
    assert second.predictions_reused == 3
    assert second.scores_created == 0
    assert second.scores_reused == 3
    assert len(state.prediction_history()) == 3
    assert len(state.score_history()) == 3


def test_backfill_enforces_safety_bound_and_hour_boundaries(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    runner = ProductionBackfillRunner(
        config=config,
        state_store=state,
        model_resolver=_FakeResolver(),
    )

    with pytest.raises(ValueError, match="configured maximum"):
        runner.run(
            start="2025-01-01T05:00:00Z",
            end="2025-01-01T08:00:00Z",
            maximum_hours=2,
        )
    with pytest.raises(ValueError, match="exactly on UTC hours"):
        runner.run(
            start="2025-01-01T05:30:00Z",
            end="2025-01-01T08:00:00Z",
        )


def test_backfill_refuses_to_cross_or_precede_the_resolved_model_epoch(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    state.record_deployment(
        ModelDeployment(
            model_name="taxi-demand",
            model_version="1",
            model_alias="champion",
            model_run_id="run-1",
            deployed_at="2025-01-01T06:00:00+00:00",
            training_start="2024-01-01T00:00:00+00:00",
            training_end="2024-12-31T00:00:00+00:00",
            data_fingerprint="data-1",
            reason="test",
        )
    )
    runner = ProductionBackfillRunner(
        config=config,
        state_store=state,
        model_resolver=_FakeResolver(),
    )

    with pytest.raises(ValueError, match="crosses a recorded model deployment"):
        runner.run(
            start="2025-01-01T05:00:00Z",
            end="2025-01-01T08:00:00Z",
        )

    state.record_deployment(
        ModelDeployment(
            model_name="taxi-demand",
            model_version="2",
            model_alias="champion",
            model_run_id="run-2",
            deployed_at="2025-01-01T09:00:00+00:00",
            training_start="2024-01-02T00:00:00+00:00",
            training_end="2025-01-01T00:00:00+00:00",
            data_fingerprint="data-2",
            reason="test",
        )
    )
    with pytest.raises(ValueError, match="does not match the model deployed"):
        runner.run(
            start="2025-01-01T10:00:00Z",
            end="2025-01-01T11:00:00Z",
        )
