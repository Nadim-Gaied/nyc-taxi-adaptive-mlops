from __future__ import annotations

from pathlib import Path

import yaml

from taxi_mlops.policies import load_lifecycle_policy


def test_frozen_lifecycle_policy_has_safe_access_and_promotion_gates() -> None:
    project_root = Path(__file__).resolve().parents[1]
    policy = load_lifecycle_policy(project_root / "configs/lifecycle_policy_v1.yaml")

    assert policy["production_candidate"]["default_training_history"] == "recent_365d"
    assert policy["model_pool"]["evaluate_before_retraining"] is True
    assert policy["promotion"]["required_prediction_coverage"] == 1.0
    assert policy["performance_monitor"]["data_drift_can_trigger_retraining_alone"] is False
    assert policy["access_policy"]["policy_validation_opened"] is False
    assert policy["access_policy"]["locked_replay_opened"] is False


def test_final_lifecycle_policy_records_validation_and_keeps_replay_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    policy = load_lifecycle_policy(
        project_root / "configs/lifecycle_policy_final_v1.yaml"
    )

    assert policy["status"] == "replay_frozen_after_policy_validation"
    assert policy["access_policy"]["policy_validation_complete"] is True
    assert policy["policy_validation_evidence"]["thresholds_changed_after_validation"] is False
    assert policy["access_policy"]["locked_replay_opened"] is False


def test_policy_validator_rejects_drift_only_retraining(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "configs/lifecycle_policy_v1.yaml"
    policy = yaml.safe_load(source.read_text(encoding="utf-8"))
    policy["performance_monitor"]["data_drift_can_trigger_retraining_alone"] = True
    invalid = tmp_path / "invalid_policy.yaml"
    invalid.write_text(yaml.safe_dump(policy), encoding="utf-8")

    try:
        load_lifecycle_policy(invalid)
    except ValueError as error:
        assert "data drift" in str(error)
    else:  # pragma: no cover - explicit failure message is clearer than bare assert
        raise AssertionError("drift-only retraining policy should have been rejected")
