import lightgbm as lgb
import mlflow
import pandas as pd
import pytest

from taxi_mlops.production import (
    ModelDeployment,
    ProductionStateStore,
    bootstrap_production_reference,
    load_production_config,
)


def _config(tmp_path):
    path = tmp_path / "production.yaml"
    path.write_text(
        """
version: 1
timezone: America/New_York
expected_zones: 2
tracking:
  uri: sqlite:///mlflow/mlflow.db
  artifact_root: mlflow/artifacts
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
  alert_window_hours: 168
  zone_lookback_days: 28
""".strip(),
        encoding="utf-8",
    )
    return load_production_config(path, project_root=tmp_path)


def _model(path) -> None:
    features = pd.DataFrame(
        {
            "pickup_zone_id": pd.Categorical([1, 2, 1, 2]),
            "lag_1h": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset = lgb.Dataset(
        features,
        label=[2.0, 3.0, 4.0, 5.0],
        categorical_feature=["pickup_zone_id"],
    )
    booster = lgb.train({"objective": "poisson", "verbosity": -1}, dataset, num_boost_round=2)
    booster.save_model(path)


def test_portable_production_bootstrap_registers_and_reuses_model(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    model_path = tmp_path / "reference.txt"
    _model(model_path)

    first = bootstrap_production_reference(
        config=config,
        state_store=state,
        model_path=model_path,
        training_start="2024-01-01T05:00:00Z",
        training_end="2025-01-01T05:00:00Z",
        archive_ref="test-v1",
    )
    second = bootstrap_production_reference(
        config=config,
        state_store=state,
        model_path=model_path,
        training_start="2024-01-01T05:00:00Z",
        training_end="2025-01-01T05:00:00Z",
        archive_ref="test-v1",
    )

    assert first.action == "registered"
    assert second.action == "reused"
    assert second.model_version == first.model_version == "1"
    assert second.model_run_id == first.model_run_id
    assert second.model_sha256 == first.model_sha256
    assert second.feature_count == 2
    assert len(state.deployment_history()) == 1

    client = mlflow.MlflowClient(tracking_uri=config.tracking.uri)
    reference = client.get_model_version_by_alias("taxi-demand", "accuracy-reference")
    champion = client.get_model_version_by_alias("taxi-demand", "champion")
    assert str(reference.version) == str(champion.version) == "1"
    assert reference.tags["model_sha256"] == first.model_sha256
    mlflow.set_tracking_uri(config.tracking.uri)
    loaded = mlflow.lightgbm.load_model("models:/taxi-demand@champion")
    assert loaded.feature_name() == ["pickup_zone_id", "lag_1h"]


def test_portable_bootstrap_refuses_existing_deployment(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    state.record_deployment(
        ModelDeployment(
            model_name="taxi-demand",
            model_version="7",
            model_alias="champion",
            model_run_id="existing-run",
            deployed_at="2025-01-01T05:00:00+00:00",
            training_start="2024-01-01T05:00:00+00:00",
            training_end="2025-01-01T05:00:00+00:00",
            data_fingerprint="another-model",
            reason="existing production model",
        )
    )
    model_path = tmp_path / "reference.txt"
    _model(model_path)

    with pytest.raises(ValueError, match="refuses to replace an existing deployment"):
        bootstrap_production_reference(
            config=config,
            state_store=state,
            model_path=model_path,
            training_start="2024-01-01T05:00:00Z",
            training_end="2025-01-01T05:00:00Z",
            archive_ref="test-v1",
        )
