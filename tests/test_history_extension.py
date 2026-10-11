from pathlib import Path

import pandas as pd
import pytest

from taxi_mlops.production.history_extension import (
    _unobservable_fall_back_hours,
    plan_historical_epochs,
)

REGISTRY = Path(__file__).resolve().parents[1] / "models/history/registry.json"


def test_portable_history_models_respect_registration_time() -> None:
    epochs = plan_historical_epochs(
        REGISTRY,
        start="2025-09-01T04:00:00Z",
        end="2025-12-24T05:00:00Z",
    )
    assert len(epochs) == 5
    assert epochs[0].start.isoformat() == "2025-09-01T04:00:00+00:00"
    assert epochs[-1].end.isoformat() == "2025-12-24T05:00:00+00:00"
    assert all(
        epoch.end == next_epoch.start for epoch, next_epoch in zip(epochs, epochs[1:], strict=False)
    )
    assert all(epoch.model_path.is_file() for epoch in epochs)
    assert all(epoch.model_path.with_suffix(".json").is_file() for epoch in epochs)


def test_history_requires_a_model_available_at_start() -> None:
    with pytest.raises(ValueError, match="no registered model"):
        plan_historical_epochs(
            REGISTRY,
            start="2025-01-01T00:00:00Z",
            end="2025-01-02T00:00:00Z",
        )


def test_unobservable_fall_back_hour_is_identified() -> None:
    start = pd.Timestamp("2025-11-02T04:00:00Z")
    end = pd.Timestamp("2025-11-02T07:00:00Z")
    assert _unobservable_fall_back_hours(start, end, "America/New_York") == {
        pd.Timestamp("2025-11-02T05:00:00Z")
    }
