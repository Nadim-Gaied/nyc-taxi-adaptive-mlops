import json

import lightgbm as lgb
import pandas as pd

from taxi_mlops.tracking import import_locked_replay_to_mlflow
from taxi_mlops.tracking.fingerprints import fingerprint_files, sha256_file


def test_file_fingerprints_are_stable_and_path_sensitive(tmp_path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    assert len(sha256_file(first)) == 64
    assert fingerprint_files([first, second], root=tmp_path) == fingerprint_files(
        [second, first], root=tmp_path
    )

    second.write_text("changed", encoding="utf-8")
    assert fingerprint_files([first, second], root=tmp_path) != fingerprint_files(
        [first], root=tmp_path
    )


def test_locked_replay_mlflow_import_is_idempotent_and_registers_reference(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    replay_dir = project_root / "replay"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    for name in (
        "splits.yaml",
        "development_experiments.yaml",
        "lifecycle_policy_final_v1.yaml",
        "locked_replay_experiment_v1.yaml",
    ):
        (config_dir / name).write_text("version: 1\n", encoding="utf-8")

    strategies = [
        "never_retrain",
        "fixed_periodic_28d",
        "performance_triggered",
        "adaptive_reuse_and_retrain",
    ]
    comparison_rows = []
    for index, strategy in enumerate(strategies):
        strategy_dir = replay_dir / strategy
        strategy_dir.mkdir(parents=True)
        comparison_rows.append(
            {
                "strategy": strategy,
                "observations": 20,
                "wape": 0.2 - index * 0.01,
                "mae": 1.0,
                "training_jobs": index + 1,
            }
        )
        (strategy_dir / "summary.json").write_text("{}", encoding="utf-8")
        for name in (
            "weekly_metrics.csv",
            "tier_metrics.csv",
            "zone_metrics.csv",
            "lifecycle_audit.csv",
        ):
            pd.DataFrame({"value": [1]}).to_csv(strategy_dir / name, index=False)
        (strategy_dir / "registry.json").write_text(
            json.dumps({"models": [], "promotions": [], "active_model_id": None}),
            encoding="utf-8",
        )

    fixed_dir = replay_dir / "fixed_periodic_28d"
    model_dir = fixed_dir / "models"
    model_dir.mkdir()
    model_id = "fixed-model"
    train = lgb.Dataset(pd.DataFrame({"x": [0.0, 1.0, 2.0]}), label=[0.0, 1.0, 2.0])
    booster = lgb.train({"objective": "poisson", "verbosity": -1}, train, num_boost_round=2)
    model_path = model_dir / f"{model_id}.txt"
    booster.save_model(model_path)
    (fixed_dir / "registry.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": model_id,
                        "trained_at": "2020-01-01T00:00:00Z",
                        "training_start": "2019-01-01T00:00:00Z",
                        "training_end": "2020-01-01T00:00:00Z",
                        "history_strategy": "recent_365d",
                        "feature_set": "full_v1",
                        "artifact_path": str(model_path),
                    }
                ],
                "promotions": [],
                "active_model_id": model_id,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(comparison_rows).to_csv(replay_dir / "strategy_comparison.csv", index=False)
    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_overall_wape_strategy": "adaptive_reuse_and_retrain",
                "locked_protocol_changed": False,
            }
        ),
        encoding="utf-8",
    )

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    first = import_locked_replay_to_mlflow(
        replay_dir=replay_dir,
        tracking_uri=tracking_uri,
        experiment_name="test-locked-replay",
        archive_ref="test-v1",
        project_root=project_root,
        registered_model_name="test-taxi-demand",
        artifact_root=str(tmp_path / "artifacts"),
    )
    second = import_locked_replay_to_mlflow(
        replay_dir=replay_dir,
        tracking_uri=tracking_uri,
        experiment_name="test-locked-replay",
        archive_ref="test-v1",
        project_root=project_root,
        registered_model_name="test-taxi-demand",
        artifact_root=str(tmp_path / "artifacts"),
    )

    assert not first.reused_existing_import
    assert first.registered_model_version == "1"
    assert set(first.strategy_run_ids) == set(strategies)
    assert second.reused_existing_import
    assert second.parent_run_id == first.parent_run_id
    assert second.strategy_run_ids == first.strategy_run_ids
    assert second.registered_model_version == first.registered_model_version
