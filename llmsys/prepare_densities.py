from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from llmsys.data import (
    ExperimentConfig,
    SystemSpec,
    load_experiment_config,
    normalize_missing_config_values,
)


def prepare_system_densities(
    config: ExperimentConfig,
    spec: SystemSpec,
) -> dict[str, object]:
    frames: dict[str, pd.DataFrame] = {}
    row_maps: dict[str, dict[tuple[object, ...], int]] = {}
    config_columns: tuple[str, ...] | None = None

    for fidelity in spec.fidelities.values():
        path = config.dataset_root / spec.name / fidelity / f"{fidelity}.csv"
        frame = pd.read_csv(path)
        columns = tuple(column for column in frame.columns if column.startswith("cfg-"))
        if config_columns is None:
            config_columns = columns
        elif columns != config_columns:
            raise ValueError(f"{spec.name} fidelity {fidelity} configuration schema mismatch")
        frame = normalize_missing_config_values(frame, columns)
        frame = collapse_config_replicates(frame, columns)
        keys = list(frame.loc[:, columns].itertuples(index=False, name=None))
        frames[fidelity] = frame
        row_maps[fidelity] = {key: index for index, key in enumerate(keys)}

    if config_columns is None:
        raise ValueError(f"{spec.name} has no fidelity tables")
    common = set.intersection(*(set(mapping) for mapping in row_maps.values()))
    if not common:
        raise ValueError(f"{spec.name} has no configurations shared by all fidelities")

    ordered = sorted(common, key=repr)
    permutation = np.random.default_rng(config.density_seed).permutation(len(ordered))
    shuffled = [ordered[int(index)] for index in permutation]

    counts: dict[str, int] = {}
    for density in (20, 40, 60, 80):
        count = max(1, math.floor(len(shuffled) * density / 100))
        selected = shuffled[:count]
        counts[str(density)] = count
        for fidelity, frame in frames.items():
            indices = [row_maps[fidelity][key] for key in selected]
            subset = frame.iloc[indices].reset_index(drop=True)
            output = (
                config.processed_root
                / str(density)
                / spec.name
                / fidelity
                / f"{fidelity}.csv"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(output, index=False, lineterminator="\n")

    return {
        "intersection_size": len(common),
        "rows": counts,
        "sources": [
            str(config.dataset_root / spec.name / fidelity / f"{fidelity}.csv")
            for fidelity in spec.fidelities.values()
        ],
    }


def collapse_config_replicates(
    frame: pd.DataFrame,
    config_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not frame.duplicated(list(config_columns), keep=False).any():
        return frame.reset_index(drop=True)

    rows = []
    for _, group in frame.groupby(list(config_columns), sort=False, dropna=False):
        row = group.iloc[0].copy()
        if len(group) > 1:
            for column in frame.columns:
                if column.startswith(("obj-", "cost-")):
                    values = pd.to_numeric(group[column], errors="coerce")
                    finite = values[np.isfinite(values.to_numpy(dtype=float))]
                    if not finite.empty:
                        row[column] = float(finite.mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=frame.columns).reset_index(drop=True)


def prepare_all_densities(config: ExperimentConfig) -> dict[str, object]:
    manifest = {
        "seed": config.density_seed,
        "density_100": {
            "mode": "raw_complete",
            "nested_with_20_80": "not_guaranteed",
        },
        "systems": {
            name: prepare_system_densities(config, spec)
            for name, spec in config.systems.items()
        },
    }
    config.processed_root.mkdir(parents=True, exist_ok=True)
    path = config.processed_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare archived density subsets.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_all_densities(load_experiment_config(args.config))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
