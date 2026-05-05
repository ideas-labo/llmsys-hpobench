# Contributing a New Benchmark

This guide explains how to add a new LLM-system benchmark dataset to LLMSYS-HPOBench. It is inspired by the HPOBench step-by-step benchmark guide, but adapted to this repository's offline, tabular benchmark format.

Before starting, read:

- [`format.md`](format.md): required CSV schema, artifact references, and directory layout.
- [`llmsys_hpobench.py`](llmsys_hpobench.py): benchmark loading API and system registry.
- Existing system manuals in [`manuals/`](manuals/): examples of system-level benchmark notes.

## Placeholders

Use these names consistently in your contribution:

- `<category>`: top-level benchmark family, for example `Engine`, `RAG`, or `Agent`.
- `<system>`: unique system name used in code, for example `vLLM`, `SGLang`, `LightRAG`, or `openhands`.
- `<fidelity_name>`: one workload/fidelity combination, for example `5.0-0.5-4-50-r1`.
- `<system_manual>`: documentation file under `manuals/<category>/<system>.md`.

`<system>` must be unique across the repository.

## Step 1: Prepare Your Branch

Fork and clone the repository, then create a feature branch:

```bash
git clone https://github.com/<github-id>/llmsys-hpobench.git
cd LLMSYS-HPOBench
git checkout -b add-<system>-benchmark
```

Use `uv run python ...` for local commands. The project does not require a heavyweight package install for basic data validation and wrapper tests.

## Step 2: Add the Data Directory

Place organized data under:

```text
experiment-data/
`-- <category>/
    `-- <system>/
        `-- <fidelity_name>/
            |-- <fidelity_name>.csv
            |-- log_file/
            |   `-- id1.log
            `-- hw_file/
                `-- id1-hw.csv
```

Rules:

- Each fidelity directory must contain exactly one main CSV.
- The main CSV file stem must match the fidelity directory name.
- `log_file/` and `hw_file/` are optional, but the CSV must still contain `log-file` and `hw-file` columns.
- If an artifact is not available, leave the corresponding CSV cell blank.
- Do not place benchmark CSVs inside `log_file/` or `hw_file/`; those directories are ignored by the loader.

## Step 3: Format the CSV Columns

Every main CSV must follow [`format.md`](format.md):

```csv
ID,cfg-...,cfg-ai-...,obj-score+,obj-latency-,cost-...,hw-file,log-file
1,...,...,...,...,...,,log_file/id1.log
```

Column rules:

- `ID`: first column, unique within the CSV.
- `cfg-ai-{name}`: AI behavior parameters such as model, inference, sampling, retrieval, or agent-policy controls.
- `cfg-{name}`: non-AI system, resource, workload, environment, and runtime parameters.
- `obj-{name}+`: objective metric to maximize.
- `obj-{name}-`: objective metric to minimize.
- `cost-{name}`: resource or cost metric.
- `hw-file`: hardware artifact path relative to the fidelity directory, or blank.
- `log-file`: combined log path relative to the fidelity directory, or blank. Use titled sections such as `===== CLIENT LOG =====` and `===== SERVER LOG =====` inside the file.

Do not add `FIDELITY_*` columns to the main CSV. Fidelity values should be encoded by the fidelity directory and CSV file name, with their meaning documented in the system manual.

If you are converting raw CSVs, run the all-system normalizer:

```bash
uv run python scripts/normalize_experiment_data.py --root experiment-data
```

If your raw data needs special handling, add a focused normalizer under `scripts/`. See [`scripts/normalize_vllm.py`](scripts/normalize_vllm.py) for raw vLLM CSVs and [`scripts/normalize_sglang.py`](scripts/normalize_sglang.py) for raw SGLang JSON samples. For systems with shared lifecycle logs, also add timestamp slicing when a row should reference only the log segment for that sampled run; [`scripts/slice_vllm_server_logs.py`](scripts/slice_vllm_server_logs.py) shows how to align client and server timestamps, including server logs that cross midnight.

## Step 4: Register the System

Open [`llmsys_hpobench.py`](llmsys_hpobench.py) and add your system to `SYSTEM_REGISTRY`:

```python
SYSTEM_REGISTRY: dict[str, str] = {
    "vLLM": "Engine/vLLM",
    "SGLang": "Engine/SGLang",
    "openhands": "Agent/openhands",
    "html_rag": "RAG/html_rag",
    "LightRAG": "RAG/LightRAG",
    "naiverag": "RAG/naiverag",
    "<system>": "<category>/<system>",
}
```

The wrapper also supports runtime registration for experiments:

```python
from llmsys_hpobench import Benchmark, register_system

register_system("<system>", "<category>/<system>")
b = Benchmark(system="<system>", root="experiment-data")
```

Use code registration in a PR when the system should become a built-in benchmark.

## Step 5: Add System Documentation

Create:

```text
manuals/<category>/<system>.md
```

Include:

- System name and benchmark scope.
- Workload/fidelity factors and filename encoding.
- Configuration columns and which are AI vs non-AI.
- Objective metrics with `+` or `-` direction.
- Cost metrics.
- Artifact availability: client logs, server logs, hardware files.
- Known missing values or caveats.
- Raw data source and generation/cleaning process.

Existing examples:

- [`manuals/Engine/vLLM.md`](manuals/Engine/vLLM.md)
- [`manuals/Engine/SGLang.md`](manuals/Engine/SGLang.md)
- [`manuals/RAG/LightRAG.md`](manuals/RAG/LightRAG.md)
- [`manuals/Agent/Openhands.md`](manuals/Agent/Openhands.md)

## Step 6: Add or Update Tests

Add tests when your contribution changes loader behavior, registration, or normalization logic.

Relevant existing tests:

- [`tests/test_llmsys_hpobench.py`](tests/test_llmsys_hpobench.py): benchmark loading, registration, fidelity discovery, and evaluation behavior.
- [`tests/test_normalize_experiment_data.py`](tests/test_normalize_experiment_data.py): common data-format normalization.
- [`tests/test_normalize_vllm.py`](tests/test_normalize_vllm.py): vLLM-specific cleaning workflow.
- [`tests/test_normalize_sglang.py`](tests/test_normalize_sglang.py): SGLang-specific JSON cleaning workflow.
- [`tests/test_slice_vllm_server_logs.py`](tests/test_slice_vllm_server_logs.py): vLLM server-log slicing and timestamp alignment.

For a new built-in system, add a small synthetic fixture test that proves:

- `Benchmark(system="<system>", root=<temp experiment-data>)` resolves the registered path.
- At least one fidelity loads.
- `evaluate()` returns `perf`, `cost`, `hardware`, and `log` groups.

## Step 7: Validate the Full Dataset

Run the normalizer if needed:

```bash
uv run python scripts/normalize_experiment_data.py --root experiment-data
```

Then run the test suite:

```bash
uv run python -m unittest tests.test_llmsys_hpobench tests.test_normalize_experiment_data tests.test_normalize_vllm tests.test_normalize_sglang tests.test_slice_vllm_server_logs -v
```

Smoke-test your system:

```bash
uv run python llmsys_hpobench.py --root experiment-data --system <system> --budget 3
```

Or from Python:

```python
from pathlib import Path
from llmsys_hpobench import Benchmark

b = Benchmark(system="<system>", root="experiment-data")
X = b.get_config_space()
Z = b.get_fidelity_space()

z = Z.sample(random_state=0)
x = X.sample(fidelity=z, random_state=0)
m = b.evaluate(config=x, fidelity=z)

print(m["perf"])
print(m["cost"])
print(m["hardware"])
print(m["log"])

fidelity_dir = Path(m["fidelity"]["path"]).parent
for value in m["log"].values():
    if value:
        print((fidelity_dir / value).exists(), fidelity_dir / value)
```

## Step 8: Run a Format Checklist

Before opening a PR, verify:

- `experiment-data/<category>/<system>/` exists.
- Every fidelity directory has exactly one main CSV.
- Every main CSV file stem matches its fidelity directory name.
- The first column is `ID`.
- AI parameters use `cfg-ai-`.
- Non-AI parameters use `cfg-`.
- Objective metrics use `obj-` and end with `+` or `-`.
- Cost metrics use `cost-`.
- `hw-file` and `log-file` are present in every main CSV.
- Nonblank artifact references resolve under the fidelity directory.
- `SYSTEM_REGISTRY` includes the system.
- `manuals/<category>/<system>.md` describes the benchmark.
- Tests and smoke tests pass.

## Pull Request Notes

In your PR description, include:

- New system name and category.
- Number of fidelities and rows.
- Main objective metrics and directions.
- Whether logs and hardware artifacts are included.
- Commands you ran for normalization and validation.
- Known missing artifacts or blank columns.

If the dataset is very large, discuss storage strategy with maintainers before adding generated artifacts directly to the repository.
