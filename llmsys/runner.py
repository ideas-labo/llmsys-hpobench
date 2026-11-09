from __future__ import annotations

import argparse
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from llmsys.data import ExperimentConfig, load_dataset_bundle, load_experiment_config
from llmsys.evaluator import OfflineEvaluator, write_results
from llmsys.fallbacks import make_fallback
from llmsys.tuners.common import derived_seed


@dataclass(frozen=True)
class Job:
    fallback: str
    density: int
    seed: int
    system: str


def build_jobs(
    config: ExperimentConfig,
    *,
    fallbacks: Iterable[str] | None = None,
    densities: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    systems: Iterable[str] | None = None,
) -> list[Job]:
    selected_fallbacks = tuple(fallbacks or config.fallbacks)
    selected_densities = tuple(densities or config.densities)
    selected_seeds = tuple(seeds or config.seeds)
    selected_systems = tuple(systems or config.systems)
    return [
        Job(fallback, density, seed, system)
        for fallback in selected_fallbacks
        for density in selected_densities
        for seed in selected_seeds
        for system in selected_systems
    ]


def result_path(config: ExperimentConfig, job: Job, tuner: str) -> Path:
    return (
        config.results_root
        / job.fallback
        / str(job.density)
        / f"run{job.seed}"
        / job.system
        / tuner
        / "results.csv"
    )


def run_job(
    config: ExperimentConfig,
    job: Job,
    tuners: Iterable[str] | None = None,
) -> list[Path]:
    _limit_nested_threads()
    bundle = load_dataset_bundle(config, job.system, job.density)
    fallback = make_fallback(
        job.fallback,
        bundle.space,
        derived_seed(job.seed, f"{job.fallback}_fallback"),
    )
    outputs = []
    for tuner in tuple(tuners or config.tuners):
        evaluator = OfflineEvaluator(bundle, fallback, config.budget_seconds)
        _run_tuner(tuner, evaluator, derived_seed(job.seed, tuner))
        output = result_path(config, job, tuner)
        write_results(output, evaluator.history, bundle)
        outputs.append(output)
    return outputs


def _run_tuner(name: str, evaluator: OfflineEvaluator, seed: int) -> None:
    if name == "random_search":
        from llmsys.tuners.random_search import run_random_search

        run_random_search(evaluator, seed)
    elif name == "smac":
        from llmsys.tuners.smac import run_smac

        run_smac(evaluator, seed)
    elif name == "hebo":
        from llmsys.tuners.hebo import run_hebo

        run_hebo(evaluator, seed)
    elif name == "hyperband":
        from llmsys.tuners.hyperband import run_hyperband

        run_hyperband(evaluator, seed)
    elif name == "bohb":
        from llmsys.tuners.bohb import run_bohb

        run_bohb(evaluator, seed)
    else:
        raise ValueError(f"unknown tuner: {name}")


def run_jobs(
    config: ExperimentConfig,
    jobs: list[Job],
    tuners: Iterable[str] | None = None,
    workers: int | None = None,
) -> None:
    selected_tuners = tuple(tuners or config.tuners)
    worker_count = config.workers if workers is None else workers
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    validate_jobs(config, jobs)
    if worker_count == 1:
        for index, job in enumerate(jobs, start=1):
            run_job(config, job, selected_tuners)
            print(f"completed {index}/{len(jobs)}: {job}", flush=True)
        return

    pool = ProcessPoolExecutor(max_workers=worker_count)
    futures = {}
    try:
        for job in jobs:
            futures[pool.submit(run_job, config, job, selected_tuners)] = job
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(
                f"completed {completed}/{len(jobs)}: {futures[future]}",
                flush=True,
            )
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def validate_jobs(config: ExperimentConfig, jobs: list[Job]) -> None:
    for system, density in sorted({(job.system, job.density) for job in jobs}):
        bundle = load_dataset_bundle(config, system, density)
        for table in bundle.tables.values():
            objective_rows = table.valid_rows(bundle.spec.objective)
            if objective_rows.empty:
                raise ValueError(
                    f"{system}/{table.name}/{bundle.spec.objective} "
                    "has no finite training values"
                )
            cost_rows = table.valid_rows(bundle.spec.cost)
            if cost_rows.empty:
                raise ValueError(
                    f"{system}/{table.name}/{bundle.spec.cost} "
                    "has no finite training values"
                )
            costs = cost_rows[bundle.spec.cost].astype(float).to_numpy()
            if np.any(costs <= 0):
                raise ValueError(
                    f"{system}/{table.name}/{bundle.spec.cost} has non-positive values"
                )


def _limit_nested_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the archived case-study protocol.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/case_study/experiment_table2.yaml"),
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fallback", action="append")
    parser.add_argument("--density", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--system", action="append")
    parser.add_argument("--tuner", action="append")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config)
    jobs = build_jobs(
        config,
        fallbacks=args.fallback,
        densities=args.density,
        seeds=args.seed,
        systems=args.system,
    )
    selected_tuners = tuple(args.tuner or config.tuners)
    _validate_selection(config, jobs, selected_tuners)
    run_jobs(config, jobs, selected_tuners, workers=args.workers)


def _validate_selection(
    config: ExperimentConfig,
    jobs: list[Job],
    tuners: tuple[str, ...],
) -> None:
    for job in jobs:
        if job.fallback not in config.fallbacks:
            raise ValueError(f"fallback not configured: {job.fallback}")
        if job.density not in config.densities:
            raise ValueError(f"density not configured: {job.density}")
        if job.seed not in config.seeds:
            raise ValueError(f"seed not configured: {job.seed}")
        if job.system not in config.systems:
            raise ValueError(f"system not configured: {job.system}")
    for tuner in tuners:
        if tuner not in config.tuners:
            raise ValueError(f"tuner not configured: {tuner}")


if __name__ == "__main__":
    main()
