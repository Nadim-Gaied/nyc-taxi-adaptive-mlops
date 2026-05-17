from __future__ import annotations

from typing import Any

from taxi_mlops.production import ForecastAlertManager, ProductionStateStore


class _RecordingNotifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[dict[str, Any]] = []

    def notify(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)
        if self.fail:
            raise ConnectionError("test endpoint unavailable")


def _manager(tmp_path, notifier=None):
    state = ProductionStateStore(tmp_path / "production.db")
    manager = ForecastAlertManager(
        state_store=state,
        warning_threshold=0.14,
        severe_threshold=0.16,
        notifier=notifier,
    )
    return state, manager


def _evaluate(manager, status, value, hour, *, complete=True):
    timestamp = f"2025-01-01T{hour:02d}:00:00+00:00"
    return manager.evaluate(
        forecast_status=status,
        window_complete=complete,
        window_wape=value,
        observed_at=timestamp,
        window_end=timestamp,
    )


def test_alert_lifecycle_is_durable_deduplicated_and_recoverable(tmp_path) -> None:
    notifier = _RecordingNotifier()
    state, manager = _manager(tmp_path, notifier)

    incomplete = _evaluate(manager, "severe", 0.20, 0, complete=False)
    assert incomplete.action == "ignored_incomplete"
    assert state.alert_history() == []

    opened = _evaluate(manager, "warning", 0.15, 1)
    assert opened.action == "opened"
    incident_id = opened.incident_id
    assert incident_id is not None

    duplicate = _evaluate(manager, "warning", 0.151, 2)
    assert duplicate.action == "deduplicated"
    assert duplicate.incident_id == incident_id
    assert len(state.alert_transition_history(incident_id)) == 1

    acknowledged = manager.acknowledge(
        incident_id,
        actor="on-call",
        note="investigating",
        acknowledged_at="2025-01-01T03:00:00Z",
    )
    assert acknowledged.action == "acknowledged"
    assert state.alert_incident(incident_id).status == "acknowledged"

    ignored = _evaluate(manager, "healthy", None, 4, complete=False)
    assert ignored.action == "ignored_incomplete"
    assert state.alert_incident(incident_id).status == "acknowledged"

    escalated = _evaluate(manager, "severe", 0.18, 5)
    assert escalated.action == "escalated"
    incident = state.alert_incident(incident_id)
    assert incident.status == "open"
    assert incident.severity == "severe"
    assert incident.acknowledged_at is None

    lower_breach = _evaluate(manager, "warning", 0.15, 6)
    assert lower_breach.action == "deduplicated"
    assert state.alert_incident(incident_id).severity == "severe"

    resolved = _evaluate(manager, "healthy", 0.12, 7)
    assert resolved.action == "resolved"
    assert resolved.incident_status == "resolved"
    assert state.alert_history(active_only=True) == []

    stale = _evaluate(manager, "severe", 0.20, 6)
    assert stale.action == "ignored_stale"
    assert len(state.alert_history()) == 1

    transitions = state.alert_transition_history(incident_id)
    assert [item.action for item in transitions] == [
        "opened",
        "acknowledged",
        "escalated",
        "resolved",
    ]
    assert [payload["event"] for payload in notifier.payloads] == [
        "forecast_alert.opened",
        "forecast_alert.acknowledged",
        "forecast_alert.escalated",
        "forecast_alert.resolved",
    ]
    assert [item["status"] for item in state.alert_delivery_history(incident_id)] == [
        "delivered",
        "delivered",
        "delivered",
        "delivered",
    ]


def test_notification_failure_is_audited_without_losing_incident(tmp_path) -> None:
    notifier = _RecordingNotifier(fail=True)
    state, manager = _manager(tmp_path, notifier)

    result = _evaluate(manager, "severe", 0.20, 1)

    assert result.action == "opened"
    assert result.notification_status == "failed"
    incident = state.alert_incident(result.incident_id)
    assert incident is not None
    assert incident.status == "open"
    deliveries = state.alert_delivery_history(result.incident_id)
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "failed"
    assert "ConnectionError" in deliveries[0]["error"]
