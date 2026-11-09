from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml


Scalar = str | int | float | bool
MISSING_CATEGORY = "__MISSING__"


@dataclass(frozen=True)
class SystemSpec:
    name: str
    fidelities: dict[int, str]
    objective: str
    cost: str

    @property
    def maximize(self) -> bool:
        return self.objective.endswith("+")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path
    processed_root: Path
    results_root: Path
    densities: tuple[int, ...]
    fallbacks: tuple[str, ...]
    tuners: tuple[str, ...]
    seeds: tuple[int, ...]
    budget_seconds: float
    workers: int
    density_seed: int
    systems: dict[str, SystemSpec]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: Literal["categorical", "integer", "continuous"]
    lower: float | int | None = None
    upper: float | int | None = None
    choices: tuple[Scalar, ...] = ()

    def sample(self, rng: np.random.Generator) -> Scalar:
        if self.kind == "categorical":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "integer":
            return int(rng.integers(int(self.lower), int(self.upper) + 1))
        return float(rng.uniform(float(self.lower), float(self.upper)))


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[ParameterSpec, ...]

    def __getitem__(self, name: str) -> ParameterSpec:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def sample(self, rng: np.random.Generator) -> dict[str, Scalar]:
        return {
            parameter.name: parameter.sample(rng)
            for parameter in self.parameters
        }


@dataclass
class FidelityTable:
    name: str
    frame: pd.DataFrame
    config_columns: tuple[str, ...]
    space: SearchSpace

    def __post_init__(self) -> None:
        self._index: dict[tuple[Scalar, ...], list[int]] = {}
        for index, row in self.frame.iterrows():
            self._index.setdefault(self.key_from_row(row), []).append(int(index))

    def key_from_row(self, row: pd.Series) -> tuple[Scalar, ...]:
        return tuple(_python_scalar(row[column]) for column in self.config_columns)

    def key_from_config(self, config: dict[str, Scalar]) -> tuple[Scalar, ...]:
        return tuple(_python_scalar(config[column]) for column in self.config_columns)

    def config_at(self, position: int) -> dict[str, Scalar]:
        row = self.frame.iloc[position]
        return {
            column: _python_scalar(row[column])
            for column in self.config_columns
        }

    def exact_value(self, config: dict[str, Scalar], label: str) -> float | None:
        indices = self._index.get(self.key_from_config(config))
        if indices is None:
            return None
        values = [
            value
            for value in (_finite_float(self.frame.at[index, label]) for index in indices)
            if value is not None
        ]
        return float(np.mean(values)) if values else None

    def valid_rows(self, label: str) -> pd.DataFrame:
        numeric = pd.to_numeric(self.frame[label], errors="coerce")
        mask = np.isfinite(numeric.to_numpy(dtype=float))
        return self.frame.loc[mask].copy()


@dataclass(frozen=True)
class DatasetBundle:
    spec: SystemSpec
    tables: dict[int, FidelityTable]
    config_columns: tuple[str, ...]
    space: SearchSpace


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    systems = {
        name: SystemSpec(
            name=name,
            fidelities={int(rung): fidelity for rung, fidelity in item["fidelities"].items()},
            objective=item["objective"],
            cost=item["cost"],
        )
        for name, item in raw["systems"].items()
    }
    return ExperimentConfig(
        dataset_root=Path(raw["dataset_root"]),
        processed_root=Path(raw["processed_root"]),
        results_root=Path(raw["results_root"]),
        densities=tuple(int(value) for value in raw["densities"]),
        fallbacks=tuple(raw["fallbacks"]),
        tuners=tuple(raw["tuners"]),
        seeds=tuple(int(value) for value in raw["seeds"]),
        budget_seconds=float(raw["budget_seconds"]),
        workers=int(raw["workers"]),
        density_seed=int(raw["density_seed"]),
        systems=systems,
    )


def load_dataset_bundle(
    config: ExperimentConfig,
    system_name: str,
    density: int,
) -> DatasetBundle:
    spec = config.systems[system_name]
    if not (spec.objective.startswith("obj-") and spec.objective.endswith(("+", "-"))):
        raise ValueError(
            f"objective {spec.objective!r} for {system_name} must end with '+' or '-'"
        )
    if not spec.cost.startswith("cost-"):
        raise ValueError(f"cost {spec.cost!r} for {system_name} must start with 'cost-'")

    root = config.dataset_root if density == 100 else config.processed_root / str(density)
    frames: dict[int, tuple[str, pd.DataFrame]] = {}
    expected_columns: tuple[str, ...] | None = None
    for rung, fidelity in sorted(spec.fidelities.items()):
        path = root / system_name / fidelity / f"{fidelity}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {system_name} fidelity CSV: {path}")
        frame = pd.read_csv(path)
        config_columns = tuple(
            column for column in frame.columns if column.startswith("cfg-")
        )
        if not config_columns:
            raise ValueError(f"{system_name} fidelity {fidelity} has no cfg-* columns")
        if expected_columns is None:
            expected_columns = config_columns
        elif config_columns != expected_columns:
            raise ValueError(
                f"{system_name} fidelity {fidelity} configuration schema mismatch"
            )
        for label in (spec.objective, spec.cost):
            if label not in frame.columns:
                raise ValueError(f"{system_name} fidelity {fidelity} missing column {label}")
        frame = normalize_missing_config_values(frame, config_columns)
        frames[rung] = (fidelity, frame)

    if expected_columns is None:
        raise ValueError(f"{system_name} has no fidelity tables")
    space = _infer_search_space(
        [frame for _, frame in frames.values()],
        expected_columns,
    )
    tables = {
        rung: FidelityTable(fidelity, frame, expected_columns, space)
        for rung, (fidelity, frame) in frames.items()
    }
    return DatasetBundle(spec, tables, expected_columns, space)


def _infer_search_space(
    frames: list[pd.DataFrame],
    config_columns: tuple[str, ...],
) -> SearchSpace:
    parameters = []
    for column in config_columns:
        values = pd.concat([frame[column] for frame in frames], ignore_index=True).dropna()
        python_values = [_python_scalar(value) for value in values]
        if not python_values:
            raise ValueError(f"configuration column {column} has no values")

        if all(isinstance(value, bool) for value in python_values):
            choices = tuple(sorted(set(python_values)))
            parameters.append(ParameterSpec(column, "categorical", choices=choices))
            continue

        numeric = pd.to_numeric(pd.Series(python_values), errors="coerce")
        if numeric.notna().all():
            numeric_values = numeric.to_numpy(dtype=float)
            lower = float(np.min(numeric_values))
            upper = float(np.max(numeric_values))
            if np.all(np.equal(numeric_values, np.floor(numeric_values))):
                parameters.append(
                    ParameterSpec(column, "integer", lower=int(lower), upper=int(upper))
                )
            else:
                parameters.append(
                    ParameterSpec(column, "continuous", lower=lower, upper=upper)
                )
            continue

        choices = tuple(
            sorted(
                set(python_values),
                key=lambda value: (type(value).__name__, str(value)),
            )
        )
        parameters.append(ParameterSpec(column, "categorical", choices=choices))
    return SearchSpace(tuple(parameters))


def normalize_missing_config_values(
    frame: pd.DataFrame,
    config_columns: tuple[str, ...],
) -> pd.DataFrame:
    normalized = frame.copy()
    for column in config_columns:
        if normalized[column].isna().any():
            values = normalized[column].astype(object)
            normalized[column] = values.where(values.notna(), MISSING_CATEGORY)
    return normalized


def _python_scalar(value: object) -> Scalar:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
