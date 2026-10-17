from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from taxi_mlops.production.config import load_production_config
from taxi_mlops.production.continuous_replay import ContinuousReplay
from taxi_mlops.production.models import ResolvedModel


class _Booster:
    pandas_categorical = [[1, 2]]

    def feature_name(self) -> list[str]:
        return ["pickup_zone_id", "lag_1h"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["lag_1h"].to_numpy(dtype=float) + 1


class _Resolver:
    def resolve(self) -> ResolvedModel:
        return ResolvedModel(
            name="taxi-demand",
            version="1",
            alias="champion",
            run_id="run-1",
            source_uri="models:/taxi-demand/1",
            booster=_Booster(),
        )


def test_continuous_replay_delays_labels_and_resumes(tmp_path: Path) -> None:
    config = replace(
        load_production_config("configs/continuous_replay.yaml", project_root=tmp_path),
        expected_zones=2,
    )
    start = pd.Timestamp("2026-08-01T04:00:00Z")
    rows = [
        {"pickup_hour": hour, "pickup_zone_id": zone, "pickup_count": 10 + zone, "lag_1h": 9 + zone}
        for hour in pd.date_range(start, periods=3, freq="h")
        for zone in (1, 2)
    ]
    feature_path = config.storage.feature_root / "2026" / "features_2026-08.parquet"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(feature_path, index=False)
    replay = ContinuousReplay(
        config,
        start=start.isoformat(),
        end=(start + pd.Timedelta(hours=3)).isoformat(),
        model_resolver=_Resolver(),
    )

    first = list(replay.run(interval_seconds=0, max_ticks=1))
    assert first[0].scored_hour is None
    assert first[1].scored_hour == start.isoformat()
    assert len(replay.state.prediction_history()) == 2
    assert len(replay.state.score_history()) == 1
    assert replay.state.prediction_for_hour(first[1].forecast_hour) is not None

    resumed = list(replay.run(interval_seconds=0, max_ticks=1))
    assert resumed[-1].scored_hour == first[1].forecast_hour
    assert len(replay.state.prediction_history()) == 3
    assert len(replay.state.score_history()) == 2
