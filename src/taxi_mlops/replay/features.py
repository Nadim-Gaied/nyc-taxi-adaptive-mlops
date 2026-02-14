"""Month-partitioned feature materialization for scalable replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from taxi_mlops.features import build_feature_table
from taxi_mlops.ingestion.inventory import month_sequence


def _month_path(root: Path, month: str, prefix: str) -> Path:
    return root / month[:4] / f"{prefix}_{month}.parquet"


def build_monthly_feature_store(
    *,
    canonical_root: str | Path,
    feature_root: str | Path,
    start_month: str,
    end_month: str,
    timezone: str = "America/New_York",
    force: bool = False,
) -> pd.DataFrame:
    """Materialize leakage-safe features with cross-month history continuity."""

    canonical_root = Path(canonical_root)
    feature_root = Path(feature_root)
    rows: list[dict[str, object]] = []
    for month in month_sequence(start_month, end_month):
        source_path = _month_path(canonical_root, month, "zone_hour")
        output_path = _month_path(feature_root, month, "features")
        if not source_path.is_file():
            raise FileNotFoundError(f"missing canonical month: {source_path}")
        if output_path.is_file() and not force:
            rows.append({"month": month, "status": "skipped_existing"})
            continue

        current = pd.read_parquet(source_path)
        current["pickup_hour"] = pd.to_datetime(
            current["pickup_hour"], utc=True, errors="raise"
        )
        current_start = pd.Timestamp(current["pickup_hour"].min())
        previous_month = str(pd.Period(month, freq="M") - 1)
        previous_path = _month_path(canonical_root, previous_month, "zone_hour")
        frames: list[pd.DataFrame] = []
        if previous_path.is_file():
            lookback_start = current_start - pd.Timedelta(days=8)
            previous = pd.read_parquet(
                previous_path,
                filters=[("pickup_hour", ">=", lookback_start)],
            )
            frames.append(previous)
        frames.append(current)
        combined = pd.concat(frames, ignore_index=True)
        features = build_feature_table(combined, timezone=timezone)
        features = features.loc[features["pickup_hour"].ge(current_start)].copy()
        if len(features) != len(current):
            raise ValueError(f"feature row count differs from canonical month {month}")
        if not features["pickup_count"].reset_index(drop=True).equals(
            current.sort_values(["pickup_hour", "pickup_zone_id"])[
                "pickup_count"
            ].reset_index(drop=True)
        ):
            raise ValueError(f"feature targets differ from canonical month {month}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path, index=False)
        rows.append(
            {
                "month": month,
                "status": "written",
                "rows": int(len(features)),
                "zones": int(features["pickup_zone_id"].nunique()),
                "start": str(features["pickup_hour"].min()),
                "end": str(features["pickup_hour"].max()),
                "output": str(output_path),
            }
        )
    return pd.DataFrame(rows)
