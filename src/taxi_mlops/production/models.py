"""Resolve production models and their lineage from MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - optional installation
        raise RuntimeError(
            'MLflow is optional. Install it with: pip install -e ".[tracking]"'
        ) from error
    return mlflow


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    version: str
    alias: str
    run_id: str
    source_uri: str
    booster: Any

    @property
    def feature_columns(self) -> list[str]:
        return list(self.booster.feature_name())

    def predict(self, features: pd.DataFrame) -> pd.Series:
        matrix = features[self.feature_columns].copy()
        categorical = getattr(self.booster, "pandas_categorical", None) or []
        if categorical and "pickup_zone_id" in matrix:
            matrix["pickup_zone_id"] = pd.Categorical(
                matrix["pickup_zone_id"], categories=categorical[0]
            )
        values = self.booster.predict(matrix)
        return pd.Series(values, index=features.index, dtype=float).clip(lower=0.0)


class MLflowLightGBMResolver:
    """Resolve one immutable registered model version through an MLflow alias."""

    def __init__(self, *, tracking_uri: str, model_name: str, alias: str) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.alias = alias

    def resolve(self) -> ResolvedModel:
        mlflow = _require_mlflow()
        mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(self.model_name, self.alias)
        source_uri = f"models:/{self.model_name}/{version.version}"
        booster = mlflow.lightgbm.load_model(source_uri)
        return ResolvedModel(
            name=self.model_name,
            version=str(version.version),
            alias=self.alias,
            run_id=str(version.run_id),
            source_uri=source_uri,
            booster=booster,
        )
