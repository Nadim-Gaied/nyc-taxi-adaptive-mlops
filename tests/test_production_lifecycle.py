import lightgbm as lgb
import mlflow
import pandas as pd

from taxi_mlops.production import (
    ProductionStateStore,
    ScheduledTrainingPipeline,
    load_production_config,
)


def _write_config(tmp_path, tracking_uri: str) -> object:
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        f"""
version: 1
timezone: America/New_York
expected_zones: 2
tracking:
  uri: {tracking_uri}
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
    return load_production_config(config_path, project_root=tmp_path)


def _register_reference(tracking_uri: str, artifact_root) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment_id = client.create_experiment(
        "archive", artifact_location=artifact_root.resolve().as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    train = lgb.Dataset(
        pd.DataFrame({"pickup_zone_id": pd.Categorical([1, 2, 1]), "lag_1h": [1, 2, 3]}),
        label=[2, 3, 4],
        categorical_feature=["pickup_zone_id"],
    )
    booster = lgb.train({"objective": "poisson", "verbosity": -1}, train, num_boost_round=2)
    with mlflow.start_run() as run:
        info = mlflow.lightgbm.log_model(booster, name="model", registered_model_name="taxi-demand")
    version = str(info.registered_model_version)
    client.set_model_version_tag("taxi-demand", version, "training_start", "2019-01-01T05:00:00Z")
    client.set_model_version_tag("taxi-demand", version, "training_end", "2020-01-01T05:00:00Z")
    client.set_model_version_tag("taxi-demand", version, "model_sha256", "reference")
    client.set_registered_model_alias("taxi-demand", "accuracy-reference", version)
    assert run.info.run_id


def _write_training_features(root, start: pd.Timestamp, end: pd.Timestamp) -> None:
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    frame = pd.DataFrame(
        {
            "pickup_hour": list(hours) * 2,
            "pickup_zone_id": [1] * len(hours) + [2] * len(hours),
            "pickup_count": [
                float(index % 24 + zone) for zone in (1, 2) for index in range(len(hours))
            ],
            "lag_1h": [
                float((index - 1) % 24 + zone) for zone in (1, 2) for index in range(len(hours))
            ],
        }
    )
    local_month = frame["pickup_hour"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    for month, partition in frame.groupby(local_month):
        path = root / month[:4] / f"features_{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        partition.sort_values(["pickup_hour", "pickup_zone_id"]).to_parquet(path, index=False)


def test_scheduled_training_bootstraps_alias_and_registers_due_model(tmp_path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    _register_reference(tracking_uri, tmp_path / "archive-artifacts")
    config = _write_config(tmp_path, tracking_uri)
    state = ProductionStateStore(config.storage.state_db)
    pipeline = ScheduledTrainingPipeline(
        config=config,
        state_store=state,
        model_parameters={"n_estimators": 3, "n_jobs": 1, "verbosity": -1},
    )

    not_due = pipeline.run("2020-01-10T05:00:00Z")
    assert not_due.action == "not_due"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    assert str(client.get_model_version_by_alias("taxi-demand", "champion").version) == "1"

    cutoff = pd.Timestamp("2021-01-01T05:00:00Z")
    history_start = pd.Timestamp("2020-01-02T05:00:00Z")
    _write_training_features(config.storage.feature_root, history_start, cutoff)
    trained = pipeline.run(cutoff)
    repeated = pipeline.run(cutoff)

    assert trained.action == "trained_and_deployed"
    assert trained.model_version == "2"
    assert trained.training_rows == 365 * 24 * 2
    assert trained.technical_validation_passed
    assert repeated.action == "not_due"
    assert repeated.model_version == "2"
    assert str(client.get_model_version_by_alias("taxi-demand", "champion").version) == "2"
    assert (
        str(client.get_model_version_by_alias("taxi-demand", "accuracy-reference").version) == "1"
    )
