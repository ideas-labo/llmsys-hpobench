---
pretty_name: LLMSYS-HPOBench
license: cc-by-4.0
tags:
- benchmark
- hyperparameter-optimization
- llm-systems
- inference
- rag
- agents
- tabular
- croissant
- mlcroissant
task_categories:
- tabular-regression
language:
- en
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/records/records.parquet
- config_name: config_long
  data_files:
  - split: train
    path: data/config_long/config_long.parquet
- config_name: metrics_long
  data_files:
  - split: train
    path: data/metrics_long/metrics_long.parquet
- config_name: artifacts_index
  data_files:
  - split: train
    path: data/artifacts_index/artifacts_index.parquet
- config_name: croissant_records
  data_files:
  - split: train
    path: metadata/croissant_records.parquet
---

# LLMSYS-HPOBench

LLMSYS-HPOBench is an offline benchmark dataset for hyperparameter optimization of real-world LLM systems. It covers inference engines, RAG pipelines, and agent frameworks, with normalized tabular measurements linked to log and hardware artifacts when available.

## Project Links

- GitHub repository, benchmark loader, and contribution guide: <https://github.com/ideas-labo/llmsys-hpobench>
- Paper: <https://arxiv.org/abs/2605.08305>
- Full data archive on Zenodo: <https://zenodo.org/records/23956381>

## What Is Hosted Here

This Hugging Face dataset repository hosts the dataset card, Croissant metadata, schema documentation, the sample manifest, a small example data package, and a Hugging Face-native version of the full normalized benchmark payload.

The original directory-shaped data package is mirrored on Zenodo for citation-friendly archival downloads. In this Hugging Face repository, the same benchmark rows are exposed as Parquet tables and the many small log/hardware text files are packed into indexed `.tar.zst` artifact shards.

## Dataset Structure

The original benchmark data is organized by category and system:

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

Each original fidelity directory contains one normalized CSV and optional per-sample artifacts:

```text
{system}/{fidelity_name}/
|-- {fidelity_name}.csv
|-- log_file/
|   `-- log-{ID}.txt
`-- hw_file/
    `-- hw-{ID}.txt
```

Main CSVs follow this schema:

| Column type | Format |
|---|---|
| Row ID | `ID` |
| AI hyperparameters | `cfg-ai-{name}` |
| Non-AI hyperparameters | `cfg-{name}` |
| Objective metrics | `obj-{name}+` or `obj-{name}-` |
| Cost metrics | `cost-{name}` |
| Hardware artifact | `hw-file` |
| Combined log artifact | `log-file` |

See `format.md` for the full cleaning and naming specification.

The Hugging Face-native layout is:

```text
data/
|-- records/records.parquet
|-- config_long/config_long.parquet
|-- metrics_long/metrics_long.parquet
`-- artifacts_index/artifacts_index.parquet
artifacts/
|-- logs/*.tar.zst
`-- hardware/*.tar.zst
```

- `records.parquet`: one row per benchmark sample, with JSON fields for grouped configuration, objective, cost, hardware, and original row values.
- `config_long.parquet`: one row per configuration value, with `is_ai` marking AI vs non-AI hyperparameters.
- `metrics_long.parquet`: one row per objective, cost, or expanded hardware metric.
- `artifacts_index.parquet`: maps each sample artifact reference to a `.tar.zst` archive and member path.

## Usage

Clone the code repository and use either the Hugging Face-native Parquet/artifact layout or the original Zenodo archive:

```bash
git clone https://github.com/ideas-labo/llmsys-hpobench
cd llmsys-hpobench
# For the original loader, extract the Zenodo archive so ./experiment-data exists.
uv run python llmsys_hpobench.py --root experiment-data --system vLLM --budget 3
```

Python example:

```python
from llmsys_hpobench import Benchmark

benchmark = Benchmark(system="vLLM", root="experiment-data")
config = benchmark.get_config_space().sample(random_state=0)
fidelity = benchmark.get_fidelity_space().sample(random_state=0)
measurement = benchmark.evaluate(config=config, fidelity=fidelity)

print(measurement["perf"])
print(measurement["cost"])
print(measurement["hardware"])
```

## Included Files

- `croissant.json`: MLCommons Croissant metadata for the dataset.
- `metadata/croissant_records.csv`: sample-level manifest with system, category, fidelity, CSV, log, and hardware references.
- `metadata/croissant_records.parquet`: Parquet version of the sample-level manifest.
- `format.md`: normalized CSV schema and artifact naming rules.
- `experiment-data/README.md`: full data package structure and extraction guide.
- `data/`: HF-native Parquet tables for benchmark rows, configs, metrics, and artifact references.
- `artifacts/`: indexed `.tar.zst` shards containing full log and hardware artifacts.
- `example-data/`: a small example subset for quick local inspection.

## License

The dataset metadata and benchmark data are released under CC-BY-4.0. The code in the GitHub repository is licensed separately under GPLv3.
