from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from llmsys.data import ParameterSpec, Scalar, SearchSpace, SystemSpec
from llmsys.evaluator import Evaluation, OfflineEvaluator
from llmsys.tuners.common import objective_loss
from llmsys.tuners.hyperband import run_hyperband


@dataclass
class BOHBSampler:
    space: SearchSpace
    spec: SystemSpec
    min_points: int | None = None
    candidates: int = 64
    top_fraction: float = 0.15
    observations: dict[int, list[tuple[dict[str, Scalar], float]]] = field(
        default_factory=dict,
        init=False,
    )
    model_based_samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.min_points is None:
            self.min_points = max(8, 2 * len(self.space.parameters) + 2)

    def observe(self, result: Evaluation) -> None:
        loss = objective_loss(self.spec, result.objective)
        self.observations.setdefault(result.rung, []).append((result.config, loss))

    def sample(
        self,
        rng: np.random.Generator,
        forbidden: set[tuple[object, ...]] | None = None,
    ) -> dict[str, Scalar]:
        excluded = forbidden or set()
        selected = self._training_data()
        if selected is None:
            return self._random_unseen(rng, excluded)

        ordered = sorted(selected, key=lambda item: item[1])
        good_count = max(2, int(np.ceil(len(ordered) * self.top_fraction)))
        good_count = min(good_count, len(ordered) - 1)
        good = [item[0] for item in ordered[:good_count]]
        bad = [item[0] for item in ordered[good_count:]]

        proposals = [
            proposal
            for proposal in (
                self._draw_from_good(good, rng) for _ in range(self.candidates)
            )
            if self._key(proposal) not in excluded
        ]
        if proposals:
            best = max(
                proposals,
                key=lambda item: self._density_ratio(item, good, bad),
            )
            self.model_based_samples += 1
            return best
        return self._random_unseen(rng, excluded)

    def _random_unseen(
        self,
        rng: np.random.Generator,
        forbidden: set[tuple[object, ...]],
    ) -> dict[str, Scalar]:
        for _ in range(100):
            config = self.space.sample(rng)
            if self._key(config) not in forbidden:
                return config
        raise RuntimeError("could not sample a unique initial configuration")

    def _key(self, config: dict[str, Scalar]) -> tuple[object, ...]:
        return tuple(config[parameter.name] for parameter in self.space.parameters)

    def _training_data(self) -> list[tuple[dict[str, Scalar], float]] | None:
        for rung in sorted(self.observations, reverse=True):
            values = self.observations[rung]
            if len(values) >= int(self.min_points):
                return values
        return None

    def _draw_from_good(
        self,
        good: list[dict[str, Scalar]],
        rng: np.random.Generator,
    ) -> dict[str, Scalar]:
        proposal: dict[str, Scalar] = {}
        for parameter in self.space.parameters:
            values = [config[parameter.name] for config in good]
            if parameter.kind == "categorical":
                probabilities = _categorical_probabilities(parameter, values)
                index = int(rng.choice(len(parameter.choices), p=probabilities))
                proposal[parameter.name] = parameter.choices[index]
                continue

            center = float(values[int(rng.integers(len(values)))])
            bandwidth = _numeric_bandwidth(parameter, values)
            sampled = float(rng.normal(center, bandwidth))
            sampled = float(np.clip(sampled, parameter.lower, parameter.upper))
            if parameter.kind == "integer":
                proposal[parameter.name] = int(round(sampled))
            else:
                proposal[parameter.name] = sampled
        return proposal

    def _density_ratio(
        self,
        config: dict[str, Scalar],
        good: list[dict[str, Scalar]],
        bad: list[dict[str, Scalar]],
    ) -> float:
        good_log = 0.0
        bad_log = 0.0
        for parameter in self.space.parameters:
            good_values = [item[parameter.name] for item in good]
            bad_values = [item[parameter.name] for item in bad]
            value = config[parameter.name]
            if parameter.kind == "categorical":
                choices = list(parameter.choices)
                index = choices.index(value)
                good_prob = _categorical_probabilities(parameter, good_values)[index]
                bad_prob = _categorical_probabilities(parameter, bad_values)[index]
            else:
                good_prob = _numeric_density(parameter, float(value), good_values)
                bad_prob = _numeric_density(parameter, float(value), bad_values)
            good_log += float(np.log(max(good_prob, 1e-300)))
            bad_log += float(np.log(max(bad_prob, 1e-300)))
        return good_log - bad_log


def _categorical_probabilities(
    parameter: ParameterSpec,
    values: list[Scalar],
) -> np.ndarray:
    counts = np.ones(len(parameter.choices), dtype=float)
    for value in values:
        counts[list(parameter.choices).index(value)] += 1.0
    return counts / counts.sum()


def _numeric_bandwidth(parameter: ParameterSpec, values: list[Scalar]) -> float:
    span = float(parameter.upper) - float(parameter.lower)
    numeric = np.asarray(values, dtype=float)
    if len(numeric) > 1:
        estimate = float(np.std(numeric, ddof=1))
    else:
        estimate = 0.0
    return max(estimate, span / np.sqrt(len(numeric)), 1e-6)


def _numeric_density(
    parameter: ParameterSpec,
    value: float,
    values: list[Scalar],
) -> float:
    bandwidth = _numeric_bandwidth(parameter, values)
    distances = (value - np.asarray(values, dtype=float)) / bandwidth
    kernels = np.exp(-0.5 * distances**2) / (np.sqrt(2 * np.pi) * bandwidth)
    return float(np.mean(kernels))


def run_bohb(evaluator: OfflineEvaluator, seed: int) -> None:
    sampler = BOHBSampler(evaluator.bundle.space, evaluator.bundle.spec)
    run_hyperband(evaluator, seed=seed, sampler=sampler)
