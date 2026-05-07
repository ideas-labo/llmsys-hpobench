# LLMSYS-HPOBench Data Cleaning Specification

This document defines the required data format, column naming rules, file naming rules, and directory layout for future LLMSYS-HPOBench data cleaning and organization.

## General Principles

- Each `.csv` file stores results for exactly one fidelity combination.
- Each row represents one evaluated configuration under that fidelity.
- The first CSV column must be `ID`, which uniquely identifies one record within the current CSV.
- CSV column names must explicitly encode the field category so that downstream cleaning, indexing, and parsing are straightforward.
- Hardware monitoring files and log files should not be expanded directly into the main CSV by default. The CSV should store a unique file name or file ID that points to the corresponding hardware and log artifacts.

## CSV Column Naming Rules

Columns should be organized in this order when possible: `ID`, hyperparameters, objective metrics, cost metrics, hardware metrics, and log info.

| Category | Column Prefix / Format | Description | Example |
|---|---|---|---|
| ID | `ID` | Row identifier within the current CSV. It may start from 1 or preserve an original traceable ID. | `1` |
| AI Hyperparameters | `cfg-ai-{parameter_name}` | AI-related configuration parameters, such as model, inference, sampling, retrieval, or agent policy parameters. | `cfg-ai-temperature` |
| Non-AI Hyperparameters | `cfg-{parameter_name}` | Non-AI configuration parameters, such as system, resource, concurrency, cache, database, or runtime parameters. | `cfg-max_num_seqs` |
| Objective Metrics | `obj-{metric_name}{+/-}` | Optimization target metrics. Use `+` for metrics to maximize and `-` for metrics to minimize. | `obj-throughput+`, `obj-TTFT-` |
| Cost Metrics | `cost-{metric_name}` | Cost or resource consumption metrics. | `cost-gpu_cache_usage`, `cost-duration` |
| Hardware File Reference | `hw-file` | File name or file ID for a hardware monitoring artifact associated with the row. Leave blank if no separate hardware artifact exists. | `hw_file/hw-1.txt` |
| Log File Reference | `log-file` | File name or file ID for the combined log artifact associated with the row. Use titled sections inside the file to distinguish client and server logs. | `log_file/log-1.txt` |

### AI vs. Non-AI Hyperparameters

Hyperparameters must distinguish AI parameters from non-AI parameters:

- AI parameters must use the `cfg-ai-` prefix.
- Non-AI parameters must use the `cfg-` prefix.

AI parameters usually include, but are not limited to:

- Model or inference policy parameters: `model`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`
- RAG or retrieval policy parameters: `retrieval_method`, `embedding_model`, `rerank_model`, `chunk_size`
- Agent or planning parameters: `planner`, `max_iterations`, `tool_policy`
- LLM behavior parameters: `enable_speculative_decoding`, `speculative_method`, `num_speculative_tokens`

Non-AI parameters usually include, but are not limited to:

- System resource parameters: `max_concurrency`, `num_workers`, `cpu_limit`, `memory_limit`
- Inference serving engineering parameters: `tp_size`, `pp_size`, `block_size`, `max_num_seqs`, `swap_space`
- Workload or environment parameters: `request_rate`, `burstiness`, `dataset_size`, `repeat`
- Cache, database, network, deployment, or other non-model behavior parameters

If a parameter is ambiguous, classify it by whether it directly changes model, retrieval, or agent intelligence behavior. If it directly changes intelligence behavior, use `cfg-ai-`; otherwise, use `cfg-`.

## CSV Example

```csv
ID,cfg-max_num_seqs,cfg-ai-enable_prefix_caching,obj-throughput+,obj-TTFT-,cost-gpu_cache_usage,cost-duration,hw-file,log-file
1,1024,True,145.2,0.62,84.7%,287.4,hw_file/hw-1.txt,log_file/log-1.txt
2,1024,False,151.6,0.58,81.3%,273.1,hw_file/hw-2.txt,log_file/log-2.txt
```

Notes:

- `obj-throughput+` means throughput is a maximization objective.
- `obj-TTFT-` means TTFT is a minimization objective.
- `hw-file` and `log-file` store artifact references. The actual files should be placed under `hw_file/` and `log_file/` in the current fidelity directory.
- If hardware monitoring metrics need to be expanded into the CSV later, expanded hardware metric columns should use the `hw-{metric_name}` prefix. The original hardware artifact reference should still remain in `hw-file`.

## CSV File Naming Rules

Each CSV file name represents one fidelity combination. The recommended pattern is:

```text
{factor1_value}-{factor2_value}-{factor3_value}.csv
```

Example:

```text
moderate-r1-memory_retrieval.csv
```

This example means:

- The first fidelity factor value is `moderate`.
- The second fidelity factor value is `r1`.
- The third fidelity factor value is `memory_retrieval`.

The file name should contain only fidelity values, not field names such as `factor1` or `factor2`. The meaning and order of the factors must be documented in the corresponding system-level or dataset-level notes.

For current normalized systems, legacy fidelity names with field labels or underscore-separated factor values are canonicalized to hyphen-separated values:

| System | Legacy Example | Canonical Example |
|---|---|---|
| OpenHands | `fc7_rc7_pd1_sc1` | `7-7-1-1` |
| LightRAG | `bridge_0_0_2` | `bridge-0-0-2` |
| NaiveRAG | `0.2_0_easy_agriculture` | `0.2-0-easy-agriculture` |
| HtmlRAG | `DC_1_HR_01_QR_05` | `1-01-05` |

If a factor value itself contains an underscore as part of a category label, keep that category label stable unless the system manual defines a safer encoding. For example, AutoGPT keeps workload categories such as `code_generation` in names like `complex-req1-code_generation`.

Do not duplicate these values as `FIDELITY_*` columns in the main CSV. The fidelity directory and CSV file name are the source of truth for fidelity values.

## Directory Layout

Each system should use the following layout:

```text
{system_name}/
`-- {fidelity_name}/
    |-- {fidelity_name}.csv
    |-- log_file/
    |   |-- log-1.txt
    |   |-- log-2.txt
    |   `-- log-3.txt
    `-- hw_file/
        |-- hw-1.txt
        |-- hw-2.txt
        `-- hw-3.txt
```

Rules:

- `{system_name}` is the system or benchmark name, for example `vLLM`, `SGLang`, `AutoGPT`, `LightRAG`, or `openhands`.
- `{fidelity_name}` must match the CSV file stem, for example `moderate-r1-memory_retrieval`.
- `log_file/` stores combined per-record log files for records under the current fidelity.
- `hw_file/` stores hardware monitoring files for records under the current fidelity.
- The log and hardware references in the CSV must uniquely resolve to files in the corresponding local directories.

## Hardware and Log File References

Hardware and log artifacts only need to be stored as unique file names or file IDs. The recommended naming pattern is:

```text
hw_file/hw-{ID}.txt
log_file/log-{ID}.txt
```

Examples:

- For `ID=1`, the hardware file is `hw_file/hw-1.txt`.
- For `ID=1`, the combined log file is `log_file/log-1.txt`.

The value stored in the CSV cell must exactly match the actual file path relative to the fidelity directory.

Combined log files should use titled sections:

```text
===== CLIENT LOG =====
source: original-client.log

...

===== SERVER LOG =====
source: original-server.log

...
```

## Cleaning Checklist

Before submitting organized data, verify that:

- The directory layout follows `{system_name}/{fidelity_name}/...`.
- Each fidelity directory contains exactly one main CSV.
- The CSV file stem matches the fidelity directory name.
- The first CSV column is `ID`.
- All AI parameters use the `cfg-ai-` prefix.
- All non-AI parameters use the `cfg-` prefix.
- All objective metrics use the `obj-` prefix and include a `+` or `-` suffix.
- All cost metrics use the `cost-` prefix.
- Hardware and log references are unique within the CSV and resolve to files under `hw_file/` and `log_file/` when the corresponding cells are not blank.
- Fidelity naming rules are consistent within each system, and the meaning of each fidelity factor is documented at the system level.

## vLLM Normalization Workflow

Raw vLLM sampling data can be normalized with:

```bash
uv run python scripts/normalize_vllm.py --source vLLM --output normalized/vLLM
```

The workflow:

- Keeps the raw `vLLM/` directory unchanged.
- Writes normalized output to `normalized/vLLM/`.
- Splits each raw vLLM CSV by `repeat`, so each output CSV corresponds to one fidelity/run combination.
- Renames fidelity directories and CSVs as `{rate}-{burstiness}-{max_concurrency}-{num_prompts}-r{repeat}`.
- Converts raw vLLM columns to the required prefixes: `cfg-ai-`, `cfg-`, `obj-`, and `cost-`.
- Adds `log-file` and `hw-file` columns for each sampled row.
- Writes `log-file` as one combined file containing the matching `client_config_{config_id}_fidelity_{fidelity_id}_*.log` and `server_config_{config_id}_*.log` sections.
- After linking, slice the server section to the client run window so each row references only the server-side log segment for that sample.
- Leaves `hw-file` empty for raw vLLM data because the available server-side artifacts are logs, not separate hardware metric files.
- Leaves any artifact reference empty when the corresponding artifact is missing.

The vLLM normalizer materializes combined `log-file` artifacts under each fidelity directory because each output log may contain content from both a client log and a server log. The legacy `--artifact-mode` option is still accepted for compatibility, but combined logs are always written as new files.

After normalization, slice vLLM server logs precisely:

```bash
uv run python scripts/slice_vllm_server_logs.py --root experiment-data/Engine/vLLM --padding-seconds 3
```

The slicer reads each `log-file`, extracts the sampling window from the client section, aligns that window to the server log timestamp timeline, and rewrites the server section to contain only matching server lines plus metadata. It handles server logs that cross midnight and safely replaces hard-linked files without mutating other links. If the server produced no lines inside the client window, the output file is still rewritten with a metadata-only server section using `selected_lines=0`.

## SGLang Normalization Workflow

Raw SGLang sampling data can be normalized in place with:

```bash
uv run python scripts/normalize_sglang.py --root experiment-data/Engine/SGLang --remove-raw
```

The workflow:

- Reads raw JSON sample outputs from directories named like `rate5.0_burst2.0_conc16_groups16_ppg8_syslen1024_qlen128_olen256`.
- Treats a missing raw `burst...` token as `burstiness=1.0`.
- Renames fidelity directories and CSVs as `{request_rate}-{burstiness}-{max_concurrency}-{gsp_num_groups}-{gsp_system_prompt_len}`, matching the SGLang factor order in `experiment-data/tab-format.tex`.
- Converts SGLang serving/runtime parameters to `cfg-*` columns.
- Converts sampling parameters such as `temperature`, `top_k`, and `top_p` to `cfg-ai-*` columns.
- Converts throughput metrics to `obj-*+` columns and latency metrics to `obj-*-` columns.
- Converts duration, token counts, and observed concurrency to `cost-*` columns.
- Adds `log-file` and `hw-file` columns for each sampled row.
- Writes `log-file` as one combined file with titled `CLIENT LOG` and `SERVER LOG` sections.
- Slices the server section to the client `Started at` / `Completed at` window with a small configurable padding. If the server log has no lines in that window, the server section remains metadata-only with `selected_lines: 0`.
- Leaves `hw-file` empty because the current raw SGLang data has no separate hardware metric artifacts.

## AutoGPT Normalization Workflow

Raw AutoGPT sampling data can be normalized in place with:

```bash
uv run python scripts/normalize_autogpt.py --root experiment-data/Agent/autogpt --remove-raw
```

If the raw `large_scale` directory is stored separately from the normalized output tree, pass it explicitly:

```bash
uv run python scripts/normalize_autogpt.py --root experiment-data/Agent/autogpt --source-root experiment-data/autogpt_original/large_scale
```

The workflow:

- Reads raw sample JSON files from `large_scale/fidelities/{task_type}_r{requests_count}_{workload_category}/`.
- Renames fidelity directories and CSVs as `{task_type}-req{requests_count}-{workload_category}`, matching the AutoGPT factor order in `experiment-data/tab-format.tex`.
- Converts agent/workload orchestration parameters to `cfg-*` columns.
- Converts LLM and agent behavior parameters such as `model_name`, `temperature`, `max_tokens`, and `use_functions_api` to `cfg-ai-*` columns.
- Converts success, correctness, adherence, and throughput metrics to `obj-*+` columns.
- Converts failure, timeout, and latency metrics to `obj-*-` columns.
- Converts duration, cycle count, token usage, and estimated cost metrics to `cost-*` columns.
- Writes raw hardware snapshots to `hw_file/hw-{ID}.txt` and stores that path in `hw-file`.
- Writes raw sample logs, `task_results` output text, and byte-range slices from external AutoGPT/vLLM server logs to `log_file/log-{ID}.txt` and stores that path in `log-file`. The source logs are resolved from the parent of `--source-root` when available, for example `experiment-data/autogpt_original/autogpt_server.log` and `experiment-data/autogpt_original/vllm.log`; the original offsets remain in the log file as provenance metadata.
- Does not duplicate `task_type`, `requests_count`, or `workload_category` as `FIDELITY_*` CSV columns.

## Benchmark Data Root

The benchmark wrapper expects organized experiment data under `experiment-data/`:

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

Built-in system registrations map system names to these paths:

| System | Registered Path |
|---|---|
| `vLLM` | `Engine/vLLM` |
| `SGLang` | `Engine/SGLang` |
| `openhands` | `Agent/openhands` |
| `autogpt` | `Agent/autogpt` |
| `html_rag` | `RAG/html_rag` |
| `LightRAG` | `RAG/LightRAG` |
| `naiverag` | `RAG/naiverag` |

Use the benchmark wrapper as:

```python
from llmsys_hpobench import Benchmark

b = Benchmark(system="vLLM", root="experiment-data")
```

New systems can be added in code without changing the directory resolver:

```python
from llmsys_hpobench import Benchmark, register_system

register_system("new_system", "NewCategory/new_system")
b = Benchmark(system="new_system", root="experiment-data")
```

For backward compatibility, `Benchmark(system="vLLM", root="experiment-data/Engine")` also works because the wrapper first checks `{root}/{system}` before consulting the registry.

## All-System Normalization Workflow

After adding or reorganizing experiment CSVs, run:

```bash
uv run python scripts/normalize_experiment_data.py --root experiment-data
```

This workflow normalizes all benchmark CSVs in place:

- Ensures the first column is `ID`.
- Adds blank `hw-file` and `log-file` columns when a system has no artifact files.
- Adds `+` or `-` direction suffixes to `obj-*` metrics.
- Removes empty CSV header columns.
- Ignores files inside `log_file/` and `hw_file/` artifact directories.
- For OpenHands, if `fc7_rc9_pd5_sc6/` contains both `fc7_rc9_pd5_sc6.csv` and `5_1__fc7_rc9_pd5_sc6.csv`, keeps the newer `5_1__fc7_rc9_pd5_sc6.csv` content and renames it to `fc7_rc9_pd5_sc6.csv`.
