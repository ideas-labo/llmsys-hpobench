from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from llmsys.data import SearchSpace
from llmsys.evaluator import OfflineEvaluator
from llmsys.tuners.common import ExternalSpaceCodec, objective_loss


def make_hebo_space(space: SearchSpace):
    cache = Path.cwd() / ".codex-work" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    from hebo.design_space.design_space import DesignSpace

    codec = ExternalSpaceCodec(space)
    parameters = []
    for parameter in space.parameters:
        if codec.uses_tokens(parameter):
            parameters.append(
                {
                    "name": parameter.name,
                    "type": "cat",
                    "categories": list(codec.tokens(parameter)),
                }
            )
        elif parameter.kind == "integer":
            parameters.append(
                {
                    "name": parameter.name,
                    "type": "int",
                    "lb": int(parameter.lower),
                    "ub": int(parameter.upper),
                }
            )
        else:
            parameters.append(
                {
                    "name": parameter.name,
                    "type": "num",
                    "lb": float(parameter.lower),
                    "ub": float(parameter.upper),
                }
            )
    return DesignSpace().parse(parameters), codec


def run_hebo(
    evaluator: OfflineEvaluator,
    seed: int,
    batch_size: int = 27,
) -> None:
    import torch
    from hebo.optimizers.hebo import HEBO

    if batch_size < 1:
        raise ValueError("HEBO batch_size must be at least 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    design_space, codec = make_hebo_space(evaluator.bundle.space)
    optimizer = HEBO(space=design_space, rand_sample=10, scramble_seed=seed)
    target_rung = max(evaluator.bundle.tables)
    evaluations = 0
    seen: set[tuple[object, ...]] = set()
    while not evaluator.exhausted:
        recommendations = optimizer.suggest(n_suggestions=batch_size)
        evaluated_in_batch = 0
        for position in range(len(recommendations)):
            recommendation = recommendations.iloc[[position]]
            config = codec.decode(recommendation.iloc[0].to_dict())
            key = tuple(
                config[column] for column in evaluator.bundle.config_columns
            )
            if key in seen:
                continue
            seen.add(key)
            result = evaluator.evaluate(config, target_rung)
            loss = objective_loss(evaluator.bundle.spec, result.objective)
            optimizer.observe(recommendation, np.asarray([[loss]], dtype=float))
            evaluations += 1
            evaluated_in_batch += 1
            if evaluator.exhausted:
                break
            if evaluations >= 1_000_000:
                raise RuntimeError("HEBO did not consume its time budget")
        if evaluated_in_batch == 0:
            raise RuntimeError("HEBO could not suggest a unique configuration")
