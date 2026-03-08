from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from taxi_mlops.production import (
    HourlyPredictionPipeline,
    HourlyScoringPipeline,
    PredictionBatch,
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
            version="7",
            alias="accuracy-reference",
            run_id="run-7",
            source_uri="models:/taxi-demand/7",
            booster=_FakeBooster(),
        )


def _write_feature_month(root, target: pd.Timestamp) -> None:
    frame = pd.DataFrame(
        {
            "pickup_hour": [target, target],
            "pickup_zone_id": [1, 2],
            "pickup_count": [11.0, 22.0],
            "lag_1h": [10.0, 20.0],
        }
    )
    month = target.strftime("%Y-%m")
    path = root / month[:4] / f"features_{month}.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)


def test_hourly_prediction_and_scoring_are_idempotent(tmp_path) -> None:
    target = pd.Timestamp("2025-01-01T05:00:00Z")
    feature_root = tmp_path / "features"
    _write_feature_month(feature_root, target)
    state = ProductionStateStore(tmp_path / "state" / "production.db")
    predictions = HourlyPredictionPipeline(
        feature_root=feature_root,
        output_root=tmp_path / "predictions",
        state_store=state,
        model_resolver=_FakeResolver(),
        timezone="America/New_York",
        expected_zones=2,
    )

    first = predictions.run(target)
    second = predictions.run(target)

    assert second == first
    assert first.row_count == 2
    persisted = pd.read_parquet(first.prediction_path)
    assert persisted["prediction"].tolist() == [11.0, 21.0]
    assert persisted["model_version"].unique().tolist() == ["7"]
    assert persisted["feature_fingerprint"].nunique() == 1

    scoring = HourlyScoringPipeline(
        feature_root=feature_root,
        output_root=tmp_path / "scores",
        state_store=state,
        timezone="America/New_York",
    )
    with pytest.raises(ValueError, match="not complete"):
        scoring.run(target, as_of=target)

    first_score = scoring.run(target, as_of=target + pd.Timedelta(hours=1))
    second_score = scoring.run(target, as_of=target + pd.Timedelta(hours=1))

    assert second_score == first_score
    assert first_score.observations == 2
    assert first_score.actual_pickups == 33.0
    assert first_score.predicted_pickups == 32.0
    assert first_score.mae == 0.5
    assert first_score.wape == pytest.approx(1 / 33)
    assert first_score.prediction_coverage == 1.0


def test_state_rejects_conflicting_lineage_for_existing_target(tmp_path) -> None:
    state = ProductionStateStore(tmp_path / "production.db")
    batch = PredictionBatch(
        batch_id="batch-1",
        target_hour="2025-01-01T05:00:00+00:00",
        generated_at="2025-01-01T04:59:00+00:00",
        model_name="taxi-demand",
        model_version="1",
        model_alias="accuracy-reference",
        model_run_id="run-1",
        model_uri="models:/taxi-demand/1",
        feature_fingerprint="features-1",
        prediction_path="predictions.parquet",
        prediction_sha256="abc",
        row_count=263,
    )
    state.record_prediction(batch)

    with pytest.raises(ValueError, match="different lineage"):
        state.record_prediction(replace(batch, batch_id="batch-2", model_version="2"))


def test_production_config_resolves_repository_relative_paths(tmp_path) -> None:
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        """
version: 1
timezone: America/New_York
expected_zones: 263
tracking:
  uri: sqlite:///state/mlflow.db
  artifact_root: state/artifacts
  experiment_name: production
  registered_model_name: taxi-demand
  production_alias: accuracy-reference
storage:
  feature_root: features
  prediction_root: predictions
  score_root: scores
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

    config = load_production_config(config_path, project_root=tmp_path)

    assert config.tracking.uri == f"sqlite:///{(tmp_path / 'state/mlflow.db').as_posix()}"
    assert config.storage.feature_root == (tmp_path / "features").resolve()
    assert config.lifecycle.retrain_every_calendar_days == 28
    assert not config.monitoring.drift_can_trigger_retraining_alone
