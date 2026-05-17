import json

import numpy as np
import pandas as pd

from taxi_mlops.production import (
    HourlyPredictionPipeline,
    HourlyScoringPipeline,
    ProductionMonitoringReporter,
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
            alias="champion",
            run_id="run-7",
            source_uri="models:/taxi-demand/7",
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
  alert_window_hours: 2
  zone_lookback_days: 28
""".strip(),
        encoding="utf-8",
    )
    return load_production_config(path, project_root=tmp_path)


def _write_features(root, targets: list[pd.Timestamp]) -> None:
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
                    "pickup_count": 22.0,
                    "lag_1h": 20.0,
                },
            ]
        )
    frame = pd.DataFrame(rows)
    month = targets[0].strftime("%Y-%m")
    path = root / month[:4] / f"features_{month}.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)


def test_monitoring_requires_complete_window_and_writes_dashboard(tmp_path) -> None:
    config = _config(tmp_path)
    targets = [
        pd.Timestamp("2025-01-01T05:00:00Z"),
        pd.Timestamp("2025-01-01T06:00:00Z"),
    ]
    _write_features(config.storage.feature_root, targets)
    state = ProductionStateStore(config.storage.state_db)
    prediction = HourlyPredictionPipeline(
        feature_root=config.storage.feature_root,
        output_root=config.storage.prediction_root,
        state_store=state,
        model_resolver=_FakeResolver(),
        expected_zones=2,
    )
    scoring = HourlyScoringPipeline(
        feature_root=config.storage.feature_root,
        output_root=config.storage.score_root,
        state_store=state,
    )
    zone_tiers = pd.DataFrame({"pickup_zone_id": [1, 2], "demand_tier": ["high", "low"]})
    reporter = ProductionMonitoringReporter(config=config, state_store=state, zone_tiers=zone_tiers)

    prediction.run(targets[0])
    scoring.run(targets[0], as_of=targets[0] + pd.Timedelta(hours=1))
    partial = reporter.build(as_of=targets[1])
    assert partial.forecast_status == "insufficient_data"
    assert not partial.alert_window_complete
    assert partial.alert_action == "ignored_incomplete"

    prediction.run(targets[1])
    scoring.run(targets[1], as_of=targets[1] + pd.Timedelta(hours=1))
    state.record_event("cycle", "started", details={"as_of": targets[1].isoformat()})
    complete = reporter.build(as_of=targets[1] + pd.Timedelta(hours=1))

    assert complete.forecast_status == "healthy"
    assert complete.operational_status == "healthy"
    assert complete.alert_window_complete
    assert complete.alert_window_wape == 2 / 66
    assert complete.alert_action == "no_active_alert"
    assert complete.active_incident_id is None
    assert config.storage.monitoring_root.joinpath("dashboard.html").is_file()
    assert config.storage.monitoring_root.joinpath("figures/zone_wape.png").is_file()
    snapshot = json.loads(config.storage.monitoring_root.joinpath("snapshot.json").read_text())
    assert snapshot["monitoring_policy"]["drift_can_trigger_retraining_alone"] is False
    assert snapshot["counts"]["scored_hours"] == 2
    assert snapshot["counts"]["active_alerts"] == 0
    assert snapshot["alert_lifecycle"]["action"] == "no_active_alert"
    tier_metrics = pd.read_csv(config.storage.monitoring_root / "tier_metrics.csv")
    assert set(tier_metrics["demand_tier"]) == {"high", "low"}
