"""Validation for frozen lifecycle-policy configuration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from taxi_mlops.datasets import split_months


def load_lifecycle_policy(path: str | Path) -> dict[str, object]:
    """Load and validate the development-frozen lifecycle policy."""

    with Path(path).open(encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    if not isinstance(policy, dict):
        raise ValueError("lifecycle policy must be a YAML mapping")
    if policy.get("version") != 1:
        raise ValueError("unsupported lifecycle policy version")
    allowed_statuses = {
        "development_frozen_pending_policy_validation",
        "replay_frozen_after_policy_validation",
    }
    if policy.get("status") not in allowed_statuses:
        raise ValueError("lifecycle policy does not have a recognized frozen status")

    monitor = policy.get("performance_monitor")
    promotion = policy.get("promotion")
    access = policy.get("access_policy")
    if not isinstance(monitor, dict) or not isinstance(promotion, dict):
        raise ValueError("lifecycle policy is missing monitor or promotion settings")
    if not isinstance(access, dict):
        raise ValueError("lifecycle policy is missing access settings")

    warning = float(monitor["weekly_wape_warning"])
    severe = float(monitor["weekly_wape_severe"])
    if not 0 < warning < severe < 1:
        raise ValueError("weekly WAPE thresholds must satisfy 0 < warning < severe < 1")
    if int(monitor["persistence_weeks"]) < 1:
        raise ValueError("performance persistence must be at least one week")
    if bool(monitor["data_drift_can_trigger_retraining_alone"]):
        raise ValueError("data drift cannot trigger retraining by itself in policy v1")

    improvement = float(promotion["minimum_relative_wape_improvement"])
    tier_regression = float(promotion["maximum_tier_wape_regression"])
    coverage = float(promotion["required_prediction_coverage"])
    if not 0 < improvement < 1:
        raise ValueError("promotion improvement must be a fraction between zero and one")
    if tier_regression < 0:
        raise ValueError("tier regression allowance cannot be negative")
    if coverage != 1.0:
        raise ValueError("policy v1 requires complete prediction coverage")
    if int(promotion["cooldown_weeks"]) < 0:
        raise ValueError("promotion cooldown cannot be negative")

    if not bool(access.get("development_complete")):
        raise ValueError("development evidence must be complete before validation")
    if bool(access.get("locked_replay_opened")):
        raise ValueError("locked replay cannot be opened by this policy configuration")
    return policy


def load_policy_validation_plan(path: str | Path) -> dict[str, object]:
    """Load the pre-registered validation plan and enforce its access boundary."""

    plan_path = Path(path)
    with plan_path.open(encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("unsupported policy-validation plan")
    if plan.get("status") != "preregistered_before_validation_access":
        raise ValueError("policy-validation plan is not pre-registered")

    access = plan.get("data_access")
    evaluation = plan.get("evaluation")
    interpretation = plan.get("interpretation_contract")
    if not all(isinstance(section, dict) for section in (access, evaluation, interpretation)):
        raise ValueError("policy-validation plan is missing required sections")
    if access["history_splits"] != ["development", "policy_validation"]:
        raise ValueError("validation history must contain only development and policy_validation")
    if access["target_split"] != "policy_validation" or access["purpose"] != "validation":
        raise ValueError("validation targets must use policy_validation access")
    if bool(access["allow_locked_replay"]) or not bool(
        interpretation["locked_replay_remains_unopened"]
    ):
        raise ValueError("policy-validation plan cannot open locked replay")
    if bool(interpretation["tune_thresholds_during_run"]):
        raise ValueError("threshold tuning during validation is prohibited")

    manifest_path = Path(plan["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = plan_path.parent.parent / manifest_path
    _, validation_start, validation_end = split_months(
        manifest_path, "policy_validation"
    )
    horizon_days = int(evaluation["horizon_calendar_days"])
    if horizon_days <= 0:
        raise ValueError("validation horizon must be positive")
    origins = evaluation.get("origins")
    if not isinstance(origins, list) or not origins:
        raise ValueError("validation origins must be a non-empty list")
    parsed_origins: list[pd.Timestamp] = []
    timezone = str(plan["timezone"])
    for value in origins:
        origin = pd.Timestamp(value)
        if origin.tzinfo is None:
            raise ValueError("validation origins must include UTC offsets")
        origin_utc = origin.tz_convert("UTC")
        end_utc = (
            origin_utc.tz_convert(timezone) + pd.DateOffset(days=horizon_days)
        ).tz_convert("UTC")
        if origin_utc < validation_start or end_utc > validation_end:
            raise ValueError("validation origin or horizon falls outside policy_validation")
        parsed_origins.append(origin_utc)
    if len(set(parsed_origins)) != len(parsed_origins):
        raise ValueError("validation origins must be unique")
    return plan
