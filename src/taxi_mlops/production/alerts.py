"""Durable forecast-alert incidents, transitions, and optional webhook delivery."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from taxi_mlops.production.config import ProductionConfig
from taxi_mlops.production.contracts import canonical_utc
from taxi_mlops.production.state import AlertIncident, ProductionStateStore

FORECAST_WAPE_ALERT = "forecast_wape"


class AlertNotifier(Protocol):
    def notify(self, payload: dict[str, Any]) -> None: ...


class WebhookAlertNotifier:
    """POST transition payloads to a user-configured HTTP(S) endpoint."""

    def __init__(self, url: str, *, timeout_seconds: float = 5.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("alert webhook must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("alert webhook timeout must be positive")
        self.url = url
        self.timeout_seconds = timeout_seconds

    def notify(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "taxi-mlops-alerts/1"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            if not 200 <= response.status < 300:
                raise RuntimeError(f"alert webhook returned HTTP {response.status}")


@dataclass(frozen=True)
class AlertEvaluation:
    action: str
    incident_id: int | None
    incident_status: str | None
    severity: str | None
    notification_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ForecastAlertManager:
    """Reconcile monitoring results with one durable WAPE incident at a time.

    This component deliberately has no dependency on training, registry aliases,
    or promotion code. An alert can notify an operator but cannot mutate a model.
    """

    def __init__(
        self,
        *,
        state_store: ProductionStateStore,
        warning_threshold: float,
        severe_threshold: float,
        notifier: AlertNotifier | None = None,
    ) -> None:
        if severe_threshold <= warning_threshold:
            raise ValueError("severe threshold must exceed warning threshold")
        self.state = state_store
        self.warning_threshold = warning_threshold
        self.severe_threshold = severe_threshold
        self.notifier = notifier

    @classmethod
    def from_config(
        cls,
        *,
        config: ProductionConfig,
        state_store: ProductionStateStore,
    ) -> ForecastAlertManager:
        notifier: AlertNotifier | None = None
        variable = config.monitoring.alert_webhook_url_env
        url = os.getenv(variable) if variable else None
        if url:
            notifier = WebhookAlertNotifier(
                url, timeout_seconds=config.monitoring.alert_webhook_timeout_seconds
            )
        return cls(
            state_store=state_store,
            warning_threshold=config.monitoring.weekly_wape_warning,
            severe_threshold=config.monitoring.weekly_wape_severe,
            notifier=notifier,
        )

    def _notify(
        self,
        *,
        action: str,
        incident: AlertIncident,
        occurred_at: str,
    ) -> str:
        if self.notifier is None:
            return "not_configured"
        payload = {
            "schema_version": 1,
            "event": f"forecast_alert.{action}",
            "occurred_at": occurred_at,
            "incident": asdict(incident),
        }
        try:
            self.notifier.notify(payload)
        except Exception as error:  # notification must not roll back durable state
            # Webhook URLs often contain credentials. Persist the exception type,
            # never a library message that could echo the configured endpoint.
            message = f"{type(error).__name__}: alert notification failed"
            self.state.record_alert_delivery(
                incident_id=incident.incident_id,
                action=action,
                attempted_at=occurred_at,
                status="failed",
                error=message,
            )
            return "failed"
        self.state.record_alert_delivery(
            incident_id=incident.incident_id,
            action=action,
            attempted_at=occurred_at,
            status="delivered",
        )
        return "delivered"

    @staticmethod
    def _result(
        action: str,
        incident: AlertIncident | None,
        notification_status: str = "not_applicable",
    ) -> AlertEvaluation:
        return AlertEvaluation(
            action=action,
            incident_id=None if incident is None else incident.incident_id,
            incident_status=None if incident is None else incident.status,
            severity=None if incident is None else incident.severity,
            notification_status=notification_status,
        )

    def evaluate(
        self,
        *,
        forecast_status: str,
        window_complete: bool,
        window_wape: float | None,
        observed_at: str,
        window_end: str,
    ) -> AlertEvaluation:
        observed = canonical_utc(observed_at, "observed_at").isoformat()
        end = canonical_utc(window_end, "window_end").isoformat()
        active = self.state.active_alert(FORECAST_WAPE_ALERT)
        latest = self.state.latest_alert(FORECAST_WAPE_ALERT)
        if latest is not None and canonical_utc(
            latest.last_observed_at, "last_observed_at"
        ) > canonical_utc(observed, "observed_at"):
            return self._result("ignored_stale", active or latest)

        if not window_complete or window_wape is None:
            return self._result("ignored_incomplete", active)
        value = float(window_wape)
        if forecast_status == "healthy":
            if active is None:
                return self._result("no_active_alert", None)
            incident = self.state.resolve_alert(
                active.incident_id,
                resolved_at=observed,
                window_end=end,
                value=value,
                threshold=self.warning_threshold,
            )
            delivery = self._notify(action="resolved", incident=incident, occurred_at=observed)
            return self._result("resolved", incident, delivery)
        if forecast_status not in {"warning", "severe"}:
            raise ValueError(f"unsupported complete-window forecast status: {forecast_status}")

        threshold = self.severe_threshold if forecast_status == "severe" else self.warning_threshold
        if active is None:
            incident, created = self.state.open_alert(
                alert_key=FORECAST_WAPE_ALERT,
                severity=forecast_status,
                observed_at=observed,
                window_end=end,
                value=value,
                threshold=threshold,
            )
            if created:
                delivery = self._notify(action="opened", incident=incident, occurred_at=observed)
                return self._result("opened", incident, delivery)
            active = incident

        if active.severity == "warning" and forecast_status == "severe":
            incident = self.state.escalate_alert(
                active.incident_id,
                observed_at=observed,
                window_end=end,
                value=value,
                threshold=threshold,
            )
            delivery = self._notify(action="escalated", incident=incident, occurred_at=observed)
            return self._result("escalated", incident, delivery)

        observed_threshold = self.severe_threshold if active.severity == "severe" else threshold
        incident = self.state.observe_alert(
            active.incident_id,
            observed_at=observed,
            window_end=end,
            value=value,
            threshold=observed_threshold,
        )
        return self._result("deduplicated", incident)

    def acknowledge(
        self,
        incident_id: int,
        *,
        actor: str,
        note: str | None = None,
        acknowledged_at: str,
    ) -> AlertEvaluation:
        actor = actor.strip()
        if not actor:
            raise ValueError("alert acknowledgement actor cannot be empty")
        occurred_at = canonical_utc(acknowledged_at, "acknowledged_at").isoformat()
        incident, changed = self.state.acknowledge_alert(
            incident_id,
            acknowledged_at=occurred_at,
            actor=actor,
            note=note,
        )
        if not changed:
            return self._result("already_acknowledged", incident)
        delivery = self._notify(action="acknowledged", incident=incident, occurred_at=occurred_at)
        return self._result("acknowledged", incident, delivery)
