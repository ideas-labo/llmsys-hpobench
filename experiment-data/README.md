# LLMSYS-HPOBench Experiment Data

This directory is the normalized data package for LLMSYS-HPOBench. It is intended to be kept at the repository root and archived on Zenodo as `experiment-data/` so users can download the benchmark data separately from the source code.

Zenodo record: <https://zenodo.org/records/20048594>

After downloading the archive from <https://zenodo.org/records/20048594>, extract it so the directory layout is:

```text
LLMSYS-HPOBench/
`-- experiment-data/
    |-- Agent/
    |-- Engine/
    `-- RAG/
```

Do not rename this directory unless you also pass the new path through the benchmark loader's `--root` argument or `Benchmark(root=...)`.

## Top-Level Layout

```text
experiment-data/
|-- Agent/
|   |-- autogpt/
|   `-- openhands/
|-- Engine/
|   |-- SGLang/
|   `-- vLLM/
`-- RAG/
    |-- html_rag/
    |-- LightRAG/
    `-- naiverag/
```

Each system directory is organized by fidelity. The fidelity factor values are encoded in the fidelity directory and CSV file name, following the order documented in `tab-format.tex` and the system manuals.

## Fidelity Directory Format

Each fidelity directory contains one main CSV named after the fidelity, plus optional per-sample artifact directories:

```text
{fidelity_name}/
|-- {fidelity_name}.csv
|-- log_file/
|   |-- log-1.txt
|   `-- log-2.txt
`-- hw_file/
    |-- hw-1.txt
    `-- hw-2.txt
```

The main CSV uses the shared LLMSYS-HPOBench schema:

| Column Type | Format |
|---|---|
| Row ID | `ID` |
| AI hyperparameters | `cfg-ai-{name}` |
| Non-AI hyperparameters | `cfg-{name}` |
| Objective metrics | `obj-{name}+` or `obj-{name}-` |
| Cost metrics | `cost-{name}` |
| Hardware artifact | `hw-file` |
| Combined log artifact | `log-file` |

`log-file` and `hw-file` values are relative paths from the fidelity directory. Empty values mean that no corresponding artifact is available for that sample.

## Artifact Files

Log artifacts use the canonical path:

```text
log_file/log-{ID}.txt
```

Hardware artifacts use the canonical path:

```text
hw_file/hw-{ID}.txt
```

When a sample has both client and server logs, the corresponding `log-{ID}.txt` file uses titled sections to keep both parts in a single artifact. The CSV row should still contain only one `log-file` column.

## Using This Data

From the repository root:

```bash
uv run python llmsys_hpobench.py --root experiment-data --system vLLM --budget 3
```

Or from Python:

```python
from llmsys_hpobench import Benchmark

benchmark = Benchmark(system="vLLM", root="experiment-data")
measurement = benchmark.evaluate(
    config=benchmark.get_config_space().sample(random_state=0),
    fidelity=benchmark.get_fidelity_space().sample(random_state=0),
)
```

The Croissant metadata at `../croissant.json` describes this data package and the sample manifest at `../metadata/croissant_records.csv`.
