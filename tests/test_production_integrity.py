import json
from pathlib import Path

import pandas as pd

from taxi_mlops.production import (
    PredictionBatch,
    ProductionIntegrityAuditor,
    ProductionStateStore,
    ScoreRecord,
    create_control_plane_checkpoint,
    load_production_config,
    verify_control_plane_checkpoint,
)
from taxi_mlops.tracking.fingerprints import sha256_file


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
""".strip(),
        encoding="utf-8",
    )
    return load_production_config(path, project_root=tmp_path)


def _populate(config, state) -> tuple[PredictionBatch, ScoreRecord]:
    target = pd.Timestamp("2025-01-01T05:00:00Z")
    batch_id = "batch-1"
    prediction_path = config.storage.prediction_root / "prediction.parquet"
    prediction_path.parent.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "pickup_hour": [target, target],
            "pickup_zone_id": [1, 2],
            "prediction": [10.0, 20.0],
            "batch_id": [batch_id, batch_id],
            "generated_at": [target.isoformat(), target.isoformat()],
            "model_name": ["taxi-demand", "taxi-demand"],
            "model_version": ["1", "1"],
            "model_alias": ["champion", "champion"],
            "model_run_id": ["run-1", "run-1"],
            "feature_fingerprint": ["features-1", "features-1"],
        }
    )
    predictions.to_parquet(prediction_path, index=False)
    batch = PredictionBatch(
        batch_id=batch_id,
        target_hour=target.isoformat(),
        generated_at=target.isoformat(),
        model_name="taxi-demand",
        model_version="1",
        model_alias="champion",
        model_run_id="run-1",
        model_uri="models:/taxi-demand/1",
        feature_fingerprint="features-1",
        prediction_path=str(prediction_path.resolve()),
        prediction_sha256=sha256_file(prediction_path),
        row_count=2,
    )
    state.record_prediction(batch)

    score_path = config.storage.score_root / "score.parquet"
    score_path.parent.mkdir(parents=True)
    scored = predictions.assign(pickup_count=[11.0, 22.0])
    scored.to_parquet(score_path, index=False)
    score = ScoreRecord(
        batch_id=batch_id,
        scored_at=(target + pd.Timedelta(hours=1)).isoformat(),
        scored_path=str(score_path.resolve()),
        observations=2,
        actual_pickups=33.0,
        predicted_pickups=30.0,
        mae=1.5,
        wape=3 / 33,
        mean_bias=-1.5,
        prediction_coverage=1.0,
        score_sha256=sha256_file(score_path),
    )
    state.record_score(score)
    return batch, score


def test_integrity_audit_and_control_plane_checkpoint(tmp_path) -> None:
    config = _config(tmp_path)
    state = ProductionStateStore(config.storage.state_db)
    _, score = _populate(config, state)
    auditor = ProductionIntegrityAuditor(config=config, state_store=state)

    passed = auditor.run(verify_registry=False)
    assert passed.status == "passed"

    checkpoint = create_control_plane_checkpoint(
        config=config,
        state_store=state,
        destination=tmp_path / "checkpoint",
    )
    manifest = json.loads(Path(checkpoint.manifest_path).read_text(encoding="utf-8"))
    assert checkpoint.prediction_artifacts == 1
    assert checkpoint.score_artifacts == 1
    assert manifest["database"]["sha256"] == checkpoint.database_sha256
    verified = verify_control_plane_checkpoint(
        config=config,
        manifest_path=checkpoint.manifest_path,
        verify_registry=False,
    )
    assert verified.status == "passed"
    assert verified.integrity_status == "passed"

    with open(score.scored_path, "ab") as stream:
        stream.write(b"tampered")
    broken_checkpoint = verify_control_plane_checkpoint(
        config=config,
        manifest_path=checkpoint.manifest_path,
        verify_registry=False,
    )
    assert broken_checkpoint.status == "failed"
    assert "checksum mismatch" in broken_checkpoint.issues[0]
    failed = auditor.run(verify_registry=False, write_report=False)
    assert failed.status == "failed"
    assert {issue.code for issue in failed.issues} == {"score_checksum_mismatch"}
