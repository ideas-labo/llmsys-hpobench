from __future__ import annotations

import tempfile
from pathlib import Path

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue

from llmsys.data import SearchSpace
from llmsys.evaluator import OfflineEvaluator
from llmsys.tuners.common import ExternalSpaceCodec, objective_loss


def make_smac_space(space: SearchSpace, seed: int) -> tuple[ConfigurationSpace, ExternalSpaceCodec]:
    codec = ExternalSpaceCodec(space)
    configspace = ConfigurationSpace(seed=seed)
    hyperparameters = []
    for parameter in space.parameters:
        if codec.uses_tokens(parameter):
            hyperparameters.append(
                CategoricalHyperparameter(parameter.name, choices=codec.tokens(parameter))
            )
        elif parameter.kind == "integer":
            hyperparameters.append(
                UniformIntegerHyperparameter(
                    parameter.name,
                    lower=int(parameter.lower),
                    upper=int(parameter.upper),
                )
            )
        else:
            hyperparameters.append(
                UniformFloatHyperparameter(
                    parameter.name,
                    lower=float(parameter.lower),
                    upper=float(parameter.upper),
                )
            )
    configspace.add(hyperparameters)
    return configspace, codec


def run_smac(evaluator: OfflineEvaluator, seed: int) -> None:
    configspace, codec = make_smac_space(evaluator.bundle.space, seed)
    target_rung = max(evaluator.bundle.tables)
    work_root = Path.cwd() / ".codex-work" / "smac"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llmsys-smac-", dir=work_root) as output_directory:
        scenario = Scenario(
            configspace,
            output_directory=output_directory,
            deterministic=True,
            n_trials=1_000_000,
            seed=seed,
        )
        initial_design = HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=30
        )
        optimizer = HyperparameterOptimizationFacade(
            scenario,
            target_function=None,
            initial_design=initial_design,
            logging_level=False,
            overwrite=True,
        )
        evaluations = 0
        while not evaluator.exhausted:
            info = optimizer.ask()
            config = codec.decode(dict(info.config))
            result = evaluator.evaluate(config, target_rung)
            optimizer.tell(
                info,
                TrialValue(
                    cost=objective_loss(evaluator.bundle.spec, result.objective),
                    time=result.cost,
                ),
                save=False,
            )
            evaluations += 1
            if evaluations >= 1_000_000:
                raise RuntimeError("SMAC did not consume its time budget")
