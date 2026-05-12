"""Small offline benchmark wrapper for LLM-system tabular datasets.

The public API mirrors the intended HPOBench-like usage:

    b = Benchmark(system="target_system")
    X = b.get_config_space()
    Z = b.get_fidelity_space()
    z = Z.sample()
    x = X.sample(fidelity=z)
    M = b.evaluate(config=x, fidelity=z)

CSV columns are grouped by prefix:
    cfg-*     -> non-AI configuration parameters
    cfg-ai-*  -> AI configuration parameters
    obj-*     -> objective/performance metrics
    cost-*    -> cost metrics
    hw-*      -> hardware metrics / hardware file identifiers
    log-*     -> log metadata / log file identifiers

Each CSV file is treated as one fidelity/environment setting. Fidelity factor
values should be encoded in the fidelity directory and CSV file name.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SYSTEM_REGISTRY: dict[str, str] = {
    "vLLM": "Engine/vLLM",
    "SGLang": "Engine/SGLang",
    "openhands": "Agent/openhands",
    "autogpt": "Agent/autogpt",
    "html_rag": "RAG/html_rag",
    "LightRAG": "RAG/LightRAG",
    "naiverag": "RAG/naiverag",
}


def register_system(system: str, relative_path: str | Path) -> None:
    """Register a system's path relative to the benchmark data root."""

    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("relative_path must be relative to the benchmark data root")
    SYSTEM_REGISTRY[system] = path.as_posix()


def registered_systems() -> dict[str, str]:
    """Return a copy of the registered system path map."""

    return dict(SYSTEM_REGISTRY)


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None

    upper = text.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False

    if text.endswith("%"):
        number_text = text[:-1].strip()
        try:
            return float(number_text)
        except ValueError:
            return text

    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _strip_prefix(column: str, prefix: str) -> str:
    return column[len(prefix) :]


def _random(random_state: int | random.Random | None) -> random.Random:
    if isinstance(random_state, random.Random):
        return random_state
    return random.Random(random_state)


def _is_same_value(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


@dataclass(frozen=True)
class Fidelity:
    """A single fidelity/environment CSV."""

    name: str
    path: Path
    values: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path), **dict(self.values)}


@dataclass
class _Record:
    fidelity: Fidelity
    config: dict[str, Any]
    config_ai: dict[str, Any]
    config_non_ai: dict[str, Any]
    perf: dict[str, Any]
    cost: dict[str, Any]
    hardware: dict[str, Any]
    log: dict[str, Any]
    row: dict[str, Any]


class Measurement(dict):
    """Evaluation result with grouped fields and a small selection helper."""

    def select(
        self,
        *,
        perf: Iterable[str] | None = None,
        cost: Iterable[str] | None = None,
        hardware: Iterable[str] | None = None,
        log: Iterable[str] | None = None,
        config: Iterable[str] | None = None,
        fidelity: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        requested = {
            "perf": perf,
            "cost": cost,
            "hardware": hardware,
            "log": log,
            "config": config,
            "fidelity": fidelity,
        }
        return {
            group: _select_fields(self[group], fields)
            for group, fields in requested.items()
            if fields is not None
        }


def _select_fields(values: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    fields = list(fields)
    if any(field is Ellipsis for field in fields):
        return dict(values)
    return {field: values[field] for field in fields if field in values}


class ConfigSpace:
    """Searchable configuration space inferred from observed CSV rows."""

    def __init__(self, benchmark: "Benchmark") -> None:
        self._benchmark = benchmark

    @property
    def columns(self) -> list[str]:
        return list(self._benchmark.config_columns)

    @property
    def ai_columns(self) -> list[str]:
        return list(self._benchmark.ai_config_columns)

    @property
    def non_ai_columns(self) -> list[str]:
        return list(self._benchmark.non_ai_config_columns)

    def sample(
        self,
        *,
        fidelity: Fidelity | str | Mapping[str, Any] | None = None,
        random_state: int | random.Random | None = None,
    ) -> dict[str, Any]:
        records = self._benchmark._records_for_fidelity(fidelity)
        if not records:
            raise ValueError("No configurations are available to sample.")
        record = _random(random_state).choice(records)
        return dict(record.config)

    def values(self, column: str) -> list[Any]:
        seen: set[Any] = set()
        values: list[Any] = []
        for record in self._benchmark.records:
            value = record.config.get(column)
            if value not in seen:
                seen.add(value)
                values.append(value)
        return values

    def describe(self) -> dict[str, dict[str, Any]]:
        description: dict[str, dict[str, Any]] = {}
        for column in self.columns:
            values = self.values(column)
            numeric_values = [value for value in values if isinstance(value, (int, float))]
            if len(numeric_values) == len(values) and values:
                description[column] = {
                    "type": "numeric",
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "values": values,
                }
            else:
                description[column] = {"type": "categorical", "values": values}
        return description


class FidelitySpace:
    """Searchable fidelity/environment space."""

    def __init__(self, fidelities: list[Fidelity]) -> None:
        self._fidelities = fidelities

    @property
    def names(self) -> list[str]:
        return [fidelity.name for fidelity in self._fidelities]

    def sample(self, *, random_state: int | random.Random | None = None) -> Fidelity:
        if not self._fidelities:
            raise ValueError("No fidelities are available to sample.")
        return _random(random_state).choice(self._fidelities)

    def get(self, name: str) -> Fidelity:
        for fidelity in self._fidelities:
            if fidelity.name == name:
                return fidelity
        raise KeyError(f"Unknown fidelity: {name}")

    def as_dicts(self) -> list[dict[str, Any]]:
        return [fidelity.as_dict() for fidelity in self._fidelities]


class Benchmark:
    """Offline tabular benchmark for one LLM system."""

    def __init__(
        self,
        system: str,
        *,
        root: str | Path = ".",
        on_missing: str = "nearest",
    ) -> None:
        if on_missing not in {"nearest", "error"}:
            raise ValueError("on_missing must be 'nearest' or 'error'.")

        self.system = system
        self.root = Path(root).resolve()
        self.system_dir = self._resolve_system_dir(system)
        self.on_missing = on_missing

        if not self.system_dir.is_dir():
            raise FileNotFoundError(f"System directory not found: {self.system_dir}")

        self.fidelities: list[Fidelity] = []
        self.records: list[_Record] = []
        self._records_by_fidelity: dict[str, list[_Record]] = {}
        self.config_columns: list[str] = []
        self.ai_config_columns: list[str] = []
        self.non_ai_config_columns: list[str] = []

        self._load()

    def get_config_space(self) -> ConfigSpace:
        return ConfigSpace(self)

    def get_fidelity_space(self) -> FidelitySpace:
        return FidelitySpace(list(self.fidelities))

    def evaluate(self, config: Mapping[str, Any], fidelity: Fidelity | str | Mapping[str, Any]) -> Measurement:
        records = self._records_for_fidelity(fidelity)
        if not records:
            raise KeyError(f"No rows found for fidelity: {fidelity!r}")

        normalized_config = self._normalize_config(config)
        record = self._find_record(records, normalized_config)
        if record is None:
            if self.on_missing == "error":
                raise KeyError(f"No exact row for config={dict(config)!r} under fidelity={fidelity!r}")
            record = self._nearest_record(records, normalized_config)

        return Measurement(
            {
                "perf": dict(record.perf),
                "cost": dict(record.cost),
                "hardware": dict(record.hardware),
                "log": dict(record.log),
                "config": dict(record.config),
                "config_ai": dict(record.config_ai),
                "config_non_ai": dict(record.config_non_ai),
                "fidelity": record.fidelity.as_dict(),
                "row": dict(record.row),
            }
        )

    def _resolve_system_dir(self, system: str) -> Path:
        direct_dir = (self.root / system).resolve()
        if direct_dir.is_dir():
            return direct_dir

        registered_path = SYSTEM_REGISTRY.get(system)
        if registered_path is not None:
            registered_dir = (self.root / registered_path).resolve()
            if registered_dir.is_dir():
                return registered_dir

        discovered = []
        if self.root.is_dir():
            for category_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
                candidate = (category_dir / system).resolve()
                if candidate.is_dir():
                    discovered.append(candidate)

        if len(discovered) == 1:
            return discovered[0]
        if len(discovered) > 1:
            choices = ", ".join(str(path) for path in discovered)
            raise ValueError(
                f"Ambiguous system name {system!r}; found multiple matches: {choices}. "
                "Register the system path explicitly or use a more specific root."
            )
        return direct_dir

    def _load(self) -> None:
        csv_paths = sorted(
            csv_path
            for csv_path in self.system_dir.rglob("*.csv")
            if self._is_benchmark_csv(csv_path)
        )
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found under: {self.system_dir}")

        for csv_path in csv_paths:
            fidelity = Fidelity(
                name=csv_path.stem,
                path=csv_path,
                values={},
            )
            records = self._read_records(csv_path, fidelity)
            if not records:
                continue
            self.fidelities.append(fidelity)
            self._records_by_fidelity[fidelity.name] = records
            self.records.extend(records)

        if not self.records:
            raise ValueError(f"No non-empty CSV rows found under: {self.system_dir}")
        if not self.config_columns or not any(record.perf or record.cost for record in self.records):
            raise ValueError(
                "No normalized benchmark columns found. Expected columns with prefixes "
                "such as cfg-*, cfg-ai-*, obj-*, cost-*, hw-*, or log-*."
            )

    def _read_records(self, csv_path: Path, fidelity: Fidelity) -> list[_Record]:
        with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            column_groups = {
                column: self._classify_column(column)
                for column in reader.fieldnames
                if column is not None
            }

            records = []
            for raw_row in reader:
                parsed_row = {column: _coerce_scalar(value or "") for column, value in raw_row.items()}
                config: dict[str, Any] = {}
                config_ai: dict[str, Any] = {}
                config_non_ai: dict[str, Any] = {}
                perf: dict[str, Any] = {}
                cost: dict[str, Any] = {}
                hardware: dict[str, Any] = {}
                log: dict[str, Any] = {}
                fidelity_values = dict(fidelity.values)

                for column, value in parsed_row.items():
                    group, key = column_groups.get(column, ("ignore", column))
                    if group == "config_ai":
                        config[key] = value
                        config_ai[key] = value
                        self._remember(self.ai_config_columns, key)
                        self._remember(self.config_columns, key)
                    elif group == "config":
                        config[key] = value
                        config_non_ai[key] = value
                        self._remember(self.non_ai_config_columns, key)
                        self._remember(self.config_columns, key)
                    elif group == "perf":
                        perf[key] = value
                    elif group == "cost":
                        cost[key] = value
                    elif group == "hardware":
                        hardware[key] = value
                    elif group == "log":
                        log[key] = value
                    elif group == "fidelity":
                        fidelity_values[key] = value

                record_fidelity = Fidelity(fidelity.name, fidelity.path, fidelity_values)
                records.append(
                    _Record(
                        fidelity=record_fidelity,
                        config=config,
                        config_ai=config_ai,
                        config_non_ai=config_non_ai,
                        perf=perf,
                        cost=cost,
                        hardware=hardware,
                        log=log,
                        row=parsed_row,
                    )
                )
            return records

    def _classify_column(self, column: str) -> tuple[str, str]:
        if column.startswith("cfg-ai-"):
            return "config_ai", _strip_prefix(column, "cfg-ai-")
        if column.startswith("cfg-"):
            return "config", _strip_prefix(column, "cfg-")
        if column.startswith("obj-"):
            return "perf", _strip_prefix(column, "obj-")
        if column.startswith("cost-"):
            return "cost", _strip_prefix(column, "cost-")
        if column.startswith("hw-"):
            return "hardware", _strip_prefix(column, "hw-")
        if column.startswith("hardware-"):
            return "hardware", _strip_prefix(column, "hardware-")
        if column.startswith("log-"):
            return "log", _strip_prefix(column, "log-")
        if column.startswith("FIDELITY_"):
            return "fidelity", _strip_prefix(column, "FIDELITY_").lower()
        return "ignore", column

    @staticmethod
    def _is_benchmark_csv(csv_path: Path) -> bool:
        artifact_dirs = {"hw_file", "log_file"}
        if csv_path.name == "sglang_multi_fidelity_benchmark_log.csv":
            return False
        return not any(part in artifact_dirs for part in csv_path.parts)

    def _records_for_fidelity(self, fidelity: Fidelity | str | Mapping[str, Any] | None) -> list[_Record]:
        if fidelity is None:
            return list(self.records)
        if isinstance(fidelity, Fidelity):
            name = fidelity.name
        elif isinstance(fidelity, str):
            name = fidelity
        else:
            name = str(fidelity.get("name") or fidelity.get("fidelity_name"))
        return list(self._records_by_fidelity.get(name, []))

    def _normalize_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in config.items():
            if key.startswith("cfg-ai-"):
                key = _strip_prefix(key, "cfg-ai-")
            elif key.startswith("cfg-"):
                key = _strip_prefix(key, "cfg-")
            normalized[key] = value
        return normalized

    def _find_record(self, records: list[_Record], config: Mapping[str, Any]) -> _Record | None:
        for record in records:
            if all(
                key in record.config and _is_same_value(record.config[key], value)
                for key, value in config.items()
            ):
                return record
        return None

    def _nearest_record(self, records: list[_Record], config: Mapping[str, Any]) -> _Record:
        return min(records, key=lambda record: self._distance(record.config, config))

    def _distance(self, candidate: Mapping[str, Any], config: Mapping[str, Any]) -> float:
        total = 0.0
        for key, value in config.items():
            candidate_value = candidate.get(key)
            if isinstance(candidate_value, (int, float)) and isinstance(value, (int, float)):
                total += abs(float(candidate_value) - float(value))
            else:
                total += 0.0 if candidate_value == value else 1.0
        return total

    @staticmethod
    def _remember(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample and evaluate an offline LLM-system benchmark dataset.")
    parser.add_argument(
        "--root",
        default="experiment-data",
        help="Dataset root containing category/system folders.",
    )
    parser.add_argument(
        "--system",
        required=True,
        help="System folder name, e.g. html_rag, LightRAG, naiverag, vLLM, SGLang, autogpt.",
    )
    parser.add_argument("--budget", type=int, default=3, help="Number of random samples to print.")
    args = parser.parse_args()

    benchmark = Benchmark(
        system=args.system,
        root=args.root,
    )
    config_space = benchmark.get_config_space()
    fidelity_space = benchmark.get_fidelity_space()
    fidelity = fidelity_space.sample(random_state=0)

    print(f"system={args.system}")
    print(f"fidelity={fidelity.name}")
    print(f"config columns={len(config_space.columns)} (ai={len(config_space.ai_columns)}, non_ai={len(config_space.non_ai_columns)})")

    rng = random.Random(0)
    budget = float(args.budget)
    t = 0.0
    index = 0
    while t < budget:
        config = config_space.sample(fidelity=fidelity, random_state=rng)
        measurement = benchmark.evaluate(config=config, fidelity=fidelity)
        cost_values = [
            value for value in measurement["cost"].values() if isinstance(value, (int, float))
        ]
        cost = sum(cost_values) if cost_values else 0.0
        print(
            {
                "iter": index,
                "config": config,
                "perf": measurement["perf"],
                "cost": measurement["cost"],
                "hardware": measurement["hardware"],
                "log": measurement["log"],
            }
        )
        t = t + cost
        index += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
