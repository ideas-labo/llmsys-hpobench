from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from llmsys.data import FidelityTable, Scalar, SearchSpace


class Fallback(Protocol):
    def predict(
        self,
        table: FidelityTable,
        label: str,
        config: dict[str, Scalar],
    ) -> float:
        ...


@dataclass
class NearestNeighborFallback:
    space: SearchSpace
    _training: dict[
        tuple[int, str], tuple[dict[str, np.ndarray], np.ndarray]
    ] = field(default_factory=dict, init=False)

    def predict(
        self,
        table: FidelityTable,
        label: str,
        config: dict[str, Scalar],
    ) -> float:
        key = (id(table), label)
        cached = self._training.get(key)
        if cached is None:
            training = table.valid_rows(label)
            if training.empty:
                raise ValueError(f"{table.name}/{label} has no finite training values")
            features = {
                parameter.name: (
                    training[parameter.name].to_numpy(dtype=object)
                    if parameter.kind == "categorical"
                    else pd.to_numeric(
                        training[parameter.name], errors="coerce"
                    ).to_numpy(dtype=float)
                )
                for parameter in self.space.parameters
            }
            targets = pd.to_numeric(training[label], errors="coerce").to_numpy(
                dtype=float
            )
            cached = (features, targets)
            self._training[key] = cached
        features, targets = cached

        distances = np.zeros(len(targets), dtype=float)
        for parameter in self.space.parameters:
            values = features[parameter.name]
            query = config[parameter.name]
            if parameter.kind == "categorical":
                distances += (values != query).astype(float)
                continue

            span = float(parameter.upper) - float(parameter.lower)
            if span == 0:
                continue
            distances += np.abs((values - float(query)) / span)

        nearest_position = int(np.argmin(distances))
        return float(targets[nearest_position])


@dataclass
class RandomForestFallback:
    space: SearchSpace
    seed: int
    n_estimators: int = 200
    _models: dict[tuple[int, str], Pipeline] = field(default_factory=dict, init=False)

    @property
    def model_count(self) -> int:
        return len(self._models)

    def predict(
        self,
        table: FidelityTable,
        label: str,
        config: dict[str, Scalar],
    ) -> float:
        key = (id(table), label)
        model = self._models.get(key)
        if model is None:
            model = self._fit(table, label)
            self._models[key] = model
        query = self._encode_categorical(
            pd.DataFrame([config], columns=table.config_columns)
        )
        return float(model.predict(query)[0])

    def _fit(self, table: FidelityTable, label: str) -> Pipeline:
        training = table.valid_rows(label)
        if training.empty:
            raise ValueError(f"{table.name}/{label} has no finite training values")

        categorical = [
            parameter.name
            for parameter in self.space.parameters
            if parameter.kind == "categorical"
        ]
        numerical = [
            parameter.name
            for parameter in self.space.parameters
            if parameter.kind != "categorical"
        ]
        transformer = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numerical", "passthrough", numerical),
            ],
            remainder="drop",
        )
        model = Pipeline(
            [
                ("features", transformer),
                (
                    "forest",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(
            self._encode_categorical(training.loc[:, table.config_columns]),
            pd.to_numeric(training[label], errors="coerce").to_numpy(dtype=float),
        )
        return model

    def _encode_categorical(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        for parameter in self.space.parameters:
            if parameter.kind == "categorical":
                encoded[parameter.name] = encoded[parameter.name].map(
                    lambda value: f"{type(value).__name__}:{value!r}"
                )
        return encoded


def make_fallback(name: str, space: SearchSpace, seed: int) -> Fallback:
    if name == "1nn":
        return NearestNeighborFallback(space)
    if name == "rf":
        return RandomForestFallback(space, seed=seed)
    raise ValueError(f"unknown fallback: {name}")
