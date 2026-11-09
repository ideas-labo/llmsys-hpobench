from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class EvaluationRecord:
    config: Mapping[str, Any]
    perf: Mapping[str, Any]
    cost: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluatedMetrics:
    perf: dict[str, float]
    cost: dict[str, float]
    source: dict[str, dict[str, str]]


@dataclass
class ExperimentalRandomForestEvaluator:
    config_columns: Sequence[str]
    reference_records: Sequence[EvaluationRecord]
    seed: int
    n_estimators: int = 200
    _models: dict[tuple[int, str, str], Pipeline] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self.config_columns = tuple(self.config_columns)
        self._categorical_columns = self._infer_categorical_columns()
        self._numerical_columns = tuple(
            column
            for column in self.config_columns
            if column not in self._categorical_columns
        )

    @property
    def model_count(self) -> int:
        return len(self._models)

    def evaluate(
        self,
        records: Sequence[EvaluationRecord],
        config: Mapping[str, Any],
    ) -> EvaluatedMetrics:
        self._validate_config(config)
        exact_records = [
            record
            for record in records
            if self._same_config(record.config, config)
        ]
        perf, perf_source = self._evaluate_group(
            records,
            exact_records,
            config,
            "perf",
        )
        cost, cost_source = self._evaluate_group(
            records,
            exact_records,
            config,
            "cost",
        )
        return EvaluatedMetrics(
            perf=perf,
            cost=cost,
            source={"perf": perf_source, "cost": cost_source},
        )

    def _evaluate_group(
        self,
        records: Sequence[EvaluationRecord],
        exact_records: Sequence[EvaluationRecord],
        config: Mapping[str, Any],
        group: str,
    ) -> tuple[dict[str, float], dict[str, str]]:
        labels = self._labels(records, group)
        values: dict[str, float] = {}
        sources: dict[str, str] = {}
        for label in labels:
            exact_values = [
                value
                for value in (
                    self._finite_float(getattr(record, group).get(label))
                    for record in exact_records
                )
                if value is not None
            ]
            if exact_values:
                values[label] = sum(exact_values) / len(exact_values)
                sources[label] = "exact"
                continue
            values[label] = self._predict(records, group, label, config)
            sources[label] = "random_forest"
        return values, sources

    def _predict(
        self,
        records: Sequence[EvaluationRecord],
        group: str,
        label: str,
        config: Mapping[str, Any],
    ) -> float:
        key = (id(records), group, label)
        model = self._models.get(key)
        if model is None:
            model = self._fit(records, group, label)
            self._models[key] = model
        query = self._encode_categorical(
            pd.DataFrame([dict(config)], columns=self.config_columns)
        )
        prediction = float(model.predict(query)[0])
        if not math.isfinite(prediction):
            raise ValueError(f"non-finite prediction for {group}/{label}")
        return prediction

    def _fit(
        self,
        records: Sequence[EvaluationRecord],
        group: str,
        label: str,
    ) -> Pipeline:
        training_records = []
        targets = []
        for record in records:
            target = self._finite_float(getattr(record, group).get(label))
            if target is None:
                continue
            training_records.append(
                {column: record.config[column] for column in self.config_columns}
            )
            targets.append(target)
        if not training_records:
            raise ValueError(f"{group}/{label} has no finite training values")

        transformer = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(self._categorical_columns),
                ),
                ("numerical", "passthrough", list(self._numerical_columns)),
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
        training = self._encode_categorical(
            pd.DataFrame(training_records, columns=self.config_columns)
        )
        model.fit(training, targets)
        return model

    def _infer_categorical_columns(self) -> tuple[str, ...]:
        categorical = []
        for column in self.config_columns:
            values = [record.config[column] for record in self.reference_records]
            if values and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                continue
            categorical.append(column)
        return tuple(categorical)

    def _encode_categorical(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        for column in self._categorical_columns:
            encoded[column] = encoded[column].map(
                lambda value: f"{type(value).__name__}:{value!r}"
            )
        return encoded

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        expected = set(self.config_columns)
        supplied = set(config)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            raise ValueError(
                f"configuration schema mismatch; missing={missing}, unknown={unknown}"
            )

    def _same_config(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> bool:
        return all(self._same_value(left[column], right[column]) for column in self.config_columns)

    @staticmethod
    def _same_value(left: Any, right: Any) -> bool:
        if isinstance(left, float) or isinstance(right, float):
            try:
                return math.isclose(
                    float(left),
                    float(right),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                return False
        return type(left) is type(right) and left == right

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _labels(
        records: Sequence[EvaluationRecord],
        group: str,
    ) -> tuple[str, ...]:
        labels: list[str] = []
        for record in records:
            for label in getattr(record, group):
                if label not in labels:
                    labels.append(label)
        return tuple(labels)
