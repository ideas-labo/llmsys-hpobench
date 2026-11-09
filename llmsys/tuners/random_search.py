from __future__ import annotations

import numpy as np

from llmsys.evaluator import OfflineEvaluator


def run_random_search(evaluator: OfflineEvaluator, seed: int) -> None:
    rng = np.random.default_rng(seed)
    target_rung = max(evaluator.bundle.tables)
    duplicate_attempts = 0
    seen: set[tuple[object, ...]] = set()
    while not evaluator.exhausted:
        config = evaluator.bundle.space.sample(rng)
        key = tuple(config[column] for column in evaluator.bundle.config_columns)
        if key in seen:
            duplicate_attempts += 1
            if duplicate_attempts >= 100:
                raise RuntimeError("random search could not find a new configuration")
            continue
        duplicate_attempts = 0
        seen.add(key)
        evaluator.evaluate(config, target_rung)
