# LLMSYS-HPOBench

LLMSYS-HPOBench is an offline benchmark and data organization project for LLM-system hyperparameter optimization. It collects sampled results from different LLM-system families, normalizes them into a shared tabular format, and exposes a lightweight Python interface for evaluating observed configurations.

The project focuses on systems where both AI parameters and non-AI system parameters matter, such as inference engines, RAG pipelines, and agent frameworks. Each benchmark row links the measured objective/cost values back to the corresponding client log, server log, and hardware artifact when those artifacts are available.

## What This Repository Provides

- A common CSV schema for LLM-system benchmark samples.
- A file layout for organizing systems, fidelities, logs, and hardware artifacts.
- A small offline benchmark wrapper in [`llmsys_hpobench.py`](llmsys_hpobench.py).
- Data normalization scripts for existing systems, including vLLM-specific log handling.
- System manuals under [`manuals/`](manuals/).
- A step-by-step contribution guide in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository Layout

```text
LLMSYS-HPOBench/
|-- llmsys_hpobench.py              # Offline benchmark API and CLI
|-- example.py                      # Minimal Python usage example
|-- analyze.py                      # Local analysis helper
|-- format.md                       # Data format and cleaning specification
|-- CONTRIBUTING.md                 # Step-by-step benchmark contribution guide
|-- manuals/                        # System-level benchmark notes
|   |-- Agent/
|   |-- Engine/
|   `-- RAG/
|-- scripts/                        # Data normalization and log-processing scripts
|   |-- normalize_experiment_data.py
|   |-- normalize_sglang.py
|   |-- normalize_vllm.py
|   `-- slice_vllm_server_logs.py
|-- tests/                          # Unit tests for loader and cleaning workflows
|-- example-data/                   # Small example dataset
`-- experiment-data/                # Local full benchmark data root
```

`experiment-data/` can be large and is ignored by Git by default. The benchmark loader expects data under this root in category/system form:

```text
experiment-data/
|-- Agent/
|   `-- openhands/
|-- Engine/
|   |-- SGLang/
|   `-- vLLM/
`-- RAG/
    |-- html_rag/
    |-- LightRAG/
    `-- naiverag/
```

Built-in system registrations currently include:

| System | Registered Path |
|---|---|
| `vLLM` | `Engine/vLLM` |
| `SGLang` | `Engine/SGLang` |
| `openhands` | `Agent/openhands` |
| `html_rag` | `RAG/html_rag` |
| `LightRAG` | `RAG/LightRAG` |
| `naiverag` | `RAG/naiverag` |

## Data Format

Each fidelity directory contains one main CSV and optional artifact folders:

```text
{system}/
`-- {fidelity_name}/
    |-- {fidelity_name}.csv
    |-- log_file/
    |   `-- id1.log
    `-- hw_file/
        `-- id1-hw.csv
```

Main CSVs use prefixed columns:

| Column Type | Format |
|---|---|
| Row ID | `ID` |
| AI hyperparameters | `cfg-ai-{name}` |
| Non-AI hyperparameters | `cfg-{name}` |
| Objective metrics | `obj-{name}+` or `obj-{name}-` |
| Cost metrics | `cost-{name}` |
| Hardware artifact | `hw-file` |
| Combined log artifact | `log-file` |

See [`format.md`](format.md) for the complete cleaning specification, including AI vs non-AI parameter rules, objective direction suffixes, artifact naming, and vLLM log slicing.

## Quick Start

The project uses only Python standard-library modules for the core loader and current tests. The commands below use `uv run python`, which is the recommended local invocation pattern for this repository.

List or sample a benchmark from the command line:

```bash
uv run python llmsys_hpobench.py --root experiment-data --system vLLM --budget 3
```

Use the benchmark wrapper from Python:

```python
from pathlib import Path
from llmsys_hpobench import Benchmark

b = Benchmark(system="vLLM", root="experiment-data")

X = b.get_config_space()
Z = b.get_fidelity_space()

z = Z.sample(random_state=0)
x = X.sample(fidelity=z, random_state=0)
m = b.evaluate(config=x, fidelity=z)

fidelity_dir = Path(m["fidelity"]["path"]).parent
log_file = fidelity_dir / m["log"]["file"]

print(m["perf"])
print(m["cost"])
print(m["hardware"])
print(log_file, log_file.exists())
```

The returned measurement groups are:

- `perf`: objective metrics from `obj-*` columns.
- `cost`: cost/resource metrics from `cost-*` columns.
- `hardware`: hardware metric columns and `hw-file`.
- `log`: `log-file`, whose target file uses titled sections for client and server logs.
- `config`: merged AI and non-AI configuration values.
- `config_ai`: values from `cfg-ai-*`.
- `config_non_ai`: values from `cfg-*`.
- `fidelity`: fidelity name and CSV path. Fidelity factor values are encoded in the fidelity directory/CSV file name, not duplicated as CSV columns.
- `row`: the original parsed CSV row.

By default, `Benchmark.evaluate()` returns the nearest observed row if an exact configuration is not found. Use `Benchmark(..., on_missing="error")` to require exact matches.

## Data Cleaning Workflows

Normalize all organized experiment CSVs in place:

```bash
uv run python scripts/normalize_experiment_data.py --root experiment-data
```

Normalize raw vLLM sampling data:

```bash
uv run python scripts/normalize_vllm.py --source vLLM --output experiment-data/Engine/vLLM --overwrite
```

Normalize raw SGLang sampling data in place:

```bash
uv run python scripts/normalize_sglang.py --root experiment-data/Engine/SGLang --remove-raw
```

The SGLang normalizer reads raw JSON result files, writes fidelity directories named as `{request_rate}-{burstiness}-{max_concurrency}-{gsp_num_groups}-{gsp_system_prompt_len}`, and materializes one combined `log-file` per sample. The server section is timestamp-sliced to the matching client run window when server lines are available; otherwise it records a metadata-only server section.

Slice normalized vLLM server logs so each row links only the server-side segment for that sampled client run:

```bash
uv run python scripts/slice_vllm_server_logs.py --root experiment-data/Engine/vLLM --padding-seconds 3
```

The vLLM slicer aligns client and server timestamps, handles logs that cross midnight, and rewrites empty windows as small metadata-only files instead of leaving full lifecycle server logs attached to individual rows.

## Adding a New System

At a high level:

1. Place data under `experiment-data/<category>/<system>/`.
2. Make every fidelity directory contain exactly one main CSV named after the fidelity.
3. Use the column prefixes defined in [`format.md`](format.md).
4. Add artifact references through `hw-file` and `log-file`.
5. Register the system in `SYSTEM_REGISTRY` in [`llmsys_hpobench.py`](llmsys_hpobench.py), or register it at runtime with `register_system()`.
6. Add a system manual under `manuals/<category>/<system>.md`.
7. Add or update tests when loader behavior or cleaning logic changes.

For the full open-source contribution process, follow [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Registering Systems Programmatically

For local experiments, you can register a system path without editing the built-in registry:

```python
from llmsys_hpobench import Benchmark, register_system

register_system("new_system", "NewCategory/new_system")
b = Benchmark(system="new_system", root="experiment-data")
```

The registered path must be relative to the benchmark data root.

## Testing

Run the current test suite:

```bash
uv run python -m unittest tests.test_llmsys_hpobench tests.test_normalize_experiment_data tests.test_normalize_vllm tests.test_normalize_sglang tests.test_slice_vllm_server_logs -v
```

Useful smoke tests:

```bash
uv run python llmsys_hpobench.py --root experiment-data --system vLLM --budget 3
uv run python example.py
```

## Documentation Map

- [`format.md`](format.md): canonical data schema, layout, and cleaning checklist.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): step-by-step guide for adding a benchmark dataset.
- [`manuals/`](manuals/): benchmark-specific notes for systems and categories.
- [`tests/`](tests/): executable examples of expected loader and normalization behavior.

## Notes for Maintainers

- Keep `format.md` as the source of truth for data layout and column naming.
- Keep `CONTRIBUTING.md` focused on external contribution steps.
- Keep `README.md` as the project overview and first-use guide.
- Do not expand large logs or hardware traces directly into main CSV files unless a future benchmark task explicitly requires it; prefer artifact references.
