from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.evaluation import (
    assign_demand_tiers,
    tier_regression_metrics,
    validate_common_targets,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(["2019-01-01T00:00:00Z", "2019-01-01T00:00:00Z"]),
            "pickup_zone_id": [1, 2],
            "pickup_count": [10, 5],
            "prediction": [9, 7],
        }
    )


def test_common_targets_accept_different_predictions_on_identical_rows() -> None:
    first = _prediction_frame()
    second = first.copy()
    second["prediction"] = [8, 6]

    targets = validate_common_targets({"first": first, "second": second})

    assert list(targets.columns) == ["pickup_hour", "pickup_zone_id", "pickup_count"]
    assert len(targets) == 2


def test_common_targets_reject_changed_actual_value() -> None:
    first = _prediction_frame()
    second = first.copy()
    second.loc[0, "pickup_count"] = 11

    with pytest.raises(ValueError, match="common target rows"):
        validate_common_targets({"first": first, "second": second})


def test_demand_tiers_are_derived_from_training_demand() -> None:
    training = pd.DataFrame(
        {
            "pickup_zone_id": [1, 2, 3, 4],
            "pickup_count": [80, 10, 5, 5],
        }
    )

    tiers = assign_demand_tiers(training).set_index("pickup_zone_id")

    assert tiers.loc[1, "demand_tier"] == "high"
    assert tiers.loc[2, "demand_tier"] == "medium"
    assert tiers.loc[3, "demand_tier"] == "medium"
    assert tiers.loc[4, "demand_tier"] == "low"
    assert tiers.loc[1, "cumulative_demand_share"] == pytest.approx(0.8)


def test_tier_metrics_use_fixed_training_assignments() -> None:
    predictions = pd.DataFrame(
        {
            "pickup_zone_id": [1, 1, 2, 2],
            "pickup_count": [10, 20, 1, 0],
            "prediction": [12, 18, 3, 2],
        }
    )
    tiers = pd.DataFrame({"pickup_zone_id": [1, 2], "demand_tier": ["high", "low"]})

    metrics = tier_regression_metrics(predictions, tiers).set_index("demand_tier")

    assert metrics.loc["high", "wape"] == pytest.approx(4 / 30)
    assert metrics.loc["low", "wape"] == pytest.approx(4 / 1)
    assert metrics.loc["high", "zones"] == 1
    assert metrics.loc["low", "prediction_coverage"] == 1.0
