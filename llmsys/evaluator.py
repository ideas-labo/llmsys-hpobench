from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llmsys.data import DatasetBundle, Scalar
from llmsys.fallbacks import Fallback


@dataclass(frozen=True)
class Evaluation:
    config: dict[str, Scalar]
    rung: int
    fidelity: str
    objective: float
    cost: float
    cumulative_cost: float


class OfflineEvaluator:
    def __init__(
        self,
        bundle: DatasetBundle,
        fallback: Fallback,
        budget_seconds: float,
    ) -> None:
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        self.bundle = bundle
        self.fallback = fallback
        self.budget_seconds = float(budget_seconds)
        self.elapsed_seconds = 0.0
        self.history: list[Evaluation] = []

    @property
    def exhausted(self) -> bool:
        return self.elapsed_seconds >= self.budget_seconds

    def evaluate(self, config: dict[str, Scalar], rung: int) -> Evaluation:
        if self.exhausted:
            raise RuntimeError("budget exhausted")
        if rung not in self.bundle.tables:
            raise ValueError(f"unknown fidelity rung: {rung}")

        table = self.bundle.tables[rung]
        objective = table.exact_value(config, self.bundle.spec.objective)
        if objective is None:
            objective = self.fallback.predict(
                table, self.bundle.spec.objective, config
            )
        cost = table.exact_value(config, self.bundle.spec.cost)
        if cost is None:
            cost = self.fallback.predict(table, self.bundle.spec.cost, config)
        if not np.isfinite(objective):
            raise ValueError(f"non-finite objective returned for {table.name}")
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError(f"invalid cost returned for {table.name}: {cost}")

        self.elapsed_seconds += float(cost)
        result = Evaluation(
            config=dict(config),
            rung=rung,
            fidelity=table.name,
            objective=float(objective),
            cost=float(cost),
            cumulative_cost=self.elapsed_seconds,
        )
        self.history.append(result)
        return result


def write_results(
    path: Path,
    history: list[Evaluation],
    bundle: DatasetBundle,
) -> None:
    columns = [
        *bundle.config_columns,
        bundle.spec.objective,
        bundle.spec.cost,
        "fidelity",
    ]
    rows = []
    for result in history:
        rows.append(
            {
                **result.config,
                bundle.spec.objective: result.objective,
                bundle.spec.cost: result.cost,
                "fidelity": result.fidelity,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
