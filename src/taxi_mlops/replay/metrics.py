"""Streaming forecast metrics for long historical replays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _Totals:
    observations: int = 0
    actual: float = 0.0
    predicted: float = 0.0
    absolute_error: float = 0.0
    signed_error: float = 0.0

    def add(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        error = predicted - actual
        self.observations += int(len(actual))
        self.actual += float(actual.sum())
        self.predicted += float(predicted.sum())
        self.absolute_error += float(np.abs(error).sum())
        self.signed_error += float(error.sum())

    def metrics(self) -> dict[str, float | int | None]:
        return {
            "observations": self.observations,
            "actual_pickups": self.actual,
            "predicted_pickups": self.predicted,
            "mae": self.absolute_error / self.observations if self.observations else None,
            "wape": self.absolute_error / self.actual if self.actual else None,
            "mean_bias": self.signed_error / self.observations if self.observations else None,
        }


class ReplayMetricsAccumulator:
    """Accumulate city, weekly, tier, and zone metrics one scored hour at a time."""

    def __init__(
        self,
        *,
        replay_start: str | pd.Timestamp,
        timezone: str,
        zone_tiers: pd.DataFrame,
    ) -> None:
        start = pd.Timestamp(replay_start)
        if start.tzinfo is None:
            raise ValueError("replay_start must include an explicit UTC offset")
        self.replay_start = start.tz_convert("UTC")
        self.timezone = timezone
        required = {"pickup_zone_id", "demand_tier"}
        missing = required.difference(zone_tiers.columns)
        if missing:
            raise ValueError(f"zone tiers are missing: {', '.join(sorted(missing))}")
        if zone_tiers["pickup_zone_id"].duplicated().any():
            raise ValueError("zone tiers contain duplicate zone IDs")
        ordered = zone_tiers.sort_values("pickup_zone_id").reset_index(drop=True)
        self.zone_ids = ordered["pickup_zone_id"].astype(int).to_numpy()
        self.zone_positions = {int(zone_id): index for index, zone_id in enumerate(self.zone_ids)}
        self.tier_labels = list(dict.fromkeys(ordered["demand_tier"].astype(str)))
        tier_position = {label: index for index, label in enumerate(self.tier_labels)}
        self.zone_tier_positions = np.array(
            [tier_position[str(value)] for value in ordered["demand_tier"]],
            dtype=int,
        )
        zone_count = len(ordered)
        self._zone_observations = np.zeros(zone_count, dtype=np.int64)
        self._zone_actual = np.zeros(zone_count, dtype=float)
        self._zone_predicted = np.zeros(zone_count, dtype=float)
        self._zone_absolute = np.zeros(zone_count, dtype=float)
        self._zone_signed = np.zeros(zone_count, dtype=float)
        self._overall = _Totals()
        self._weekly: dict[int, _Totals] = {}
        self._tiers = [_Totals() for _ in self.tier_labels]

    def update(self, scored: pd.DataFrame) -> None:
        required = {"pickup_hour", "pickup_zone_id", "pickup_count", "prediction"}
        missing = required.difference(scored.columns)
        if missing:
            raise ValueError(f"scored predictions are missing: {', '.join(sorted(missing))}")
        if scored.empty:
            return
        actual = scored["pickup_count"].to_numpy(dtype=float)
        predicted = scored["prediction"].to_numpy(dtype=float)
        error = predicted - actual
        self._overall.add(actual, predicted)

        try:
            positions = np.array(
                [self.zone_positions[int(value)] for value in scored["pickup_zone_id"]],
                dtype=int,
            )
        except KeyError as error_key:
            raise ValueError(f"scored zone has no demand tier: {error_key.args[0]}") from error_key
        np.add.at(self._zone_observations, positions, 1)
        np.add.at(self._zone_actual, positions, actual)
        np.add.at(self._zone_predicted, positions, predicted)
        np.add.at(self._zone_absolute, positions, np.abs(error))
        np.add.at(self._zone_signed, positions, error)

        for tier_index in np.unique(self.zone_tier_positions[positions]):
            mask = self.zone_tier_positions[positions] == tier_index
            self._tiers[int(tier_index)].add(actual[mask], predicted[mask])

        local_dates = pd.to_datetime(scored["pickup_hour"], utc=True).dt.tz_convert(self.timezone)
        replay_local = self.replay_start.tz_convert(self.timezone)
        normalized_start = replay_local.normalize().tz_localize(None)
        normalized_dates = local_dates.dt.normalize().dt.tz_localize(None)
        week_indices = (normalized_dates - normalized_start).dt.days.floordiv(7).astype(int)
        for week_index in week_indices.unique():
            mask = week_indices.eq(week_index).to_numpy()
            totals = self._weekly.setdefault(int(week_index), _Totals())
            totals.add(actual[mask], predicted[mask])

    def overall_metrics(self) -> dict[str, float | int | None]:
        metrics = self._overall.metrics()
        zone_mae = np.divide(
            self._zone_absolute,
            self._zone_observations,
            out=np.full_like(self._zone_absolute, np.nan),
            where=self._zone_observations > 0,
        )
        eligible = zone_mae[~np.isnan(zone_mae)]
        metrics.update(
            {
                "zones": int(len(eligible)),
                "macro_zone_mae": float(np.mean(eligible)) if len(eligible) else None,
                "zone_mae_p50": float(np.quantile(eligible, 0.50)) if len(eligible) else None,
                "zone_mae_p90": float(np.quantile(eligible, 0.90)) if len(eligible) else None,
                "zone_mae_p95": float(np.quantile(eligible, 0.95)) if len(eligible) else None,
                "zone_mae_max": float(np.max(eligible)) if len(eligible) else None,
            }
        )
        return metrics

    def weekly_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        start_local = self.replay_start.tz_convert(self.timezone)
        for week_index, totals in sorted(self._weekly.items()):
            rows.append(
                {
                    "week_index": week_index + 1,
                    "week_start": str((start_local + pd.DateOffset(days=7 * week_index)).date()),
                    **totals.metrics(),
                }
            )
        return pd.DataFrame(rows)

    def tier_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"demand_tier": label, **totals.metrics()}
                for label, totals in zip(self.tier_labels, self._tiers, strict=True)
            ]
        )

    def zone_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pickup_zone_id": self.zone_ids,
                "observations": self._zone_observations,
                "actual_pickups": self._zone_actual,
                "predicted_pickups": self._zone_predicted,
                "mae": np.divide(
                    self._zone_absolute,
                    self._zone_observations,
                    out=np.full_like(self._zone_absolute, np.nan),
                    where=self._zone_observations > 0,
                ),
                "wape": np.divide(
                    self._zone_absolute,
                    self._zone_actual,
                    out=np.full_like(self._zone_absolute, np.nan),
                    where=self._zone_actual > 0,
                ),
                "mean_bias": np.divide(
                    self._zone_signed,
                    self._zone_observations,
                    out=np.full_like(self._zone_signed, np.nan),
                    where=self._zone_observations > 0,
                ),
            }
        )
