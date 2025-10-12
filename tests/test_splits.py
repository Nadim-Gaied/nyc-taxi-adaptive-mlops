from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.datasets import load_named_split, split_months
from taxi_mlops.ingestion.yellow_taxi import DataContractError


def _write_manifest(path) -> None:
    path.write_text(
        """
development:
  start: "2019-01-01T00:00:00-05:00"
  end: "2019-03-01T00:00:00-05:00"
policy_validation:
  start: "2019-03-01T00:00:00-05:00"
  end: "2019-04-01T00:00:00-04:00"
locked_replay:
  start: "2020-01-01T00:00:00-05:00"
  end: "2020-02-01T00:00:00-05:00"
""".strip(),
        encoding="utf-8",
    )


def test_split_months_uses_end_exclusive_boundaries(tmp_path) -> None:
    manifest = tmp_path / "splits.yaml"
    _write_manifest(manifest)

    months, start, end = split_months(manifest, "development")

    assert months == ["2019-01", "2019-02"]
    assert str(start) == "2019-01-01 05:00:00+00:00"
    assert str(end) == "2019-03-01 05:00:00+00:00"


def test_locked_replay_rejects_eda_access(tmp_path) -> None:
    manifest = tmp_path / "splits.yaml"
    _write_manifest(manifest)

    with pytest.raises(DataContractError, match="cannot be loaded for eda"):
        load_named_split(
            monthly_root=tmp_path,
            manifest_path=manifest,
            split_name="locked_replay",
            purpose="eda",
        )


def test_loader_requires_every_month_in_split(tmp_path) -> None:
    manifest = tmp_path / "splits.yaml"
    _write_manifest(manifest)
    month_dir = tmp_path / "2019"
    month_dir.mkdir()
    pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(["2019-01-01T05:00:00Z"]),
            "pickup_zone_id": [1],
            "pickup_count": [1],
        }
    ).to_parquet(month_dir / "zone_hour_2019-01.parquet", index=False)

    with pytest.raises(DataContractError, match="missing 1 month"):
        load_named_split(
            monthly_root=tmp_path,
            manifest_path=manifest,
            split_name="development",
            purpose="eda",
        )

