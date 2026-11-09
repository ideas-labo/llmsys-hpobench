from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from llmsys.data import Scalar, SearchSpace
from llmsys.evaluator import Evaluation, OfflineEvaluator


class CandidateSampler(Protocol):
    def sample(
        self,
        rng: np.random.Generator,
        forbidden: set[tuple[object, ...]] | None = None,
    ) -> dict[str, Scalar]:
        ...

    def observe(self, result: Evaluation) -> None:
        ...


@dataclass
class RandomCandidateSampler:
    space: SearchSpace

    def sample(
        self,
        rng: np.random.Generator,
        forbidden: set[tuple[object, ...]] | None = None,
    ) -> dict[str, Scalar]:
        return self.space.sample(rng)

    def observe(self, result: Evaluation) -> None:
        return None


def run_hyperband(
    evaluator: OfflineEvaluator,
    seed: int,
    sampler: CandidateSampler | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    candidate_sampler = sampler or RandomCandidateSampler(evaluator.bundle.space)
    bracket_order = (2, 1, 0)
    seen_initial: set[tuple[object, ...]] = set()
    while not evaluator.exhausted:
        for bracket in bracket_order:
            if evaluator.exhausted:
                return
            initial_count = int(np.ceil(3 / (bracket + 1) * (2**bracket)))
            first_rung = 4 // (2**bracket)
            configs = []
            for _ in range(initial_count):
                for _attempt in range(100):
                    config = candidate_sampler.sample(rng, seen_initial)
                    key = tuple(
                        config[column] for column in evaluator.bundle.config_columns
                    )
                    if key not in seen_initial:
                        seen_initial.add(key)
                        configs.append(config)
                        break
                else:
                    raise RuntimeError("could not sample a unique initial configuration")

            for stage in range(bracket + 1):
                rung = first_rung * (2**stage)
                evaluated: list[Evaluation] = []
                for config in configs:
                    if evaluator.exhausted:
                        return
                    result = evaluator.evaluate(config, rung)
                    candidate_sampler.observe(result)
                    evaluated.append(result)

                if stage < bracket:
                    evaluated.sort(
                        key=lambda item: item.objective,
                        reverse=evaluator.bundle.spec.maximize,
                    )
                    keep = max(1, len(evaluated) // 2)
                    configs = [item.config for item in evaluated[:keep]]
