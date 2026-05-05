"""Normalize raw vLLM sampling CSVs into the project data format."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


ArtifactMode = Literal["reference", "copy", "hardlink"]

VLLM_CSV_RE = re.compile(
    r"^rate(?P<rate>[0-9.]+)_burst(?P<burst>[0-9.]+)_conc(?P<conc>[0-9]+)_prompts(?P<prompts>[0-9]+)\.csv$"
)
CLIENT_LOG_RE = re.compile(
    r"^client_config_(?P<config_id>[^_]+)_fidelity_(?P<fidelity_id>[0-9]+)_(?P<stamp>.+)\.log$"
)
SERVER_LOG_RE = re.compile(r"^server_config_(?P<config_id>[^_]+)_(?P<stamp>.+)\.log$")

AI_CONFIG_COLUMNS = {
    "enable_prefix_caching",
    "enable_speculative_decoding",
    "speculative_method",
    "num_speculative_tokens",
    "prompt_lookup_max",
    "temperature",
    "top_k",
    "min_p",
    "repetition_penalty",
    "length_penalty",
}

NON_AI_CONFIG_COLUMNS = {
    "config_id",
    "tp_size",
    "pp_size",
    "block_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "disable_custom_all_reduce",
    "swap_space",
    "max_seq_len_to_capture",
    "enforce_eager",
    "scheduling_policy",
}

OBJECTIVE_MAX_COLUMNS = {
    "completed",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "successful_requests",
    "mean_bleu",
}

OBJECTIVE_MIN_COLUMNS = {
    "failed_requests",
    "p95_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
}

COST_COLUMNS = {
    "benchmark_duration_s",
    "total_input",
    "total_output",
    "gpu_kv_cache_usage_avg",
    "process_cpu_seconds_avg",
}


@dataclass(frozen=True)
class VllmCsv:
    path: Path
    rate: str
    burst: str
    conc: str
    prompts: str
    rows: list[dict[str, str]]


def normalize_vllm_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    artifact_mode: ArtifactMode = "reference",
    overwrite: bool = False,
) -> dict[str, int]:
    """Normalize raw vLLM data into ``output_dir``.

    ``log-file`` links to one per-run log file. The file contains titled
    client/server sections when both sides are available. ``hw-file`` is
    reserved for separate hardware metric artifacts; raw vLLM currently has no
    such separate file, so the value is blank. Missing artifacts are represented
    by empty cells.
    """

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"vLLM source directory not found: {source}")
    if output == source or source in output.parents:
        raise ValueError("output_dir must not be the source directory or a child of it")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    csv_inputs = _read_vllm_csvs(source)
    if not csv_inputs:
        raise FileNotFoundError(f"No raw vLLM fidelity CSVs found under: {source}")

    client_logs = _index_client_logs(source / "logs")
    server_logs = _index_server_logs(source / "logs")
    value_spaces = _value_spaces(csv_inputs)

    summary = {
        "input_csv_files": len(csv_inputs),
        "output_csv_files": 0,
        "rows": 0,
        "linked_client_logs": 0,
        "missing_client_logs": 0,
        "linked_server_logs": 0,
        "missing_server_logs": 0,
        "linked_hardware_files": 0,
        "missing_hardware_files": 0,
    }

    for source_csv in csv_inputs:
        rows_by_repeat: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in source_csv.rows:
            rows_by_repeat[row.get("repeat") or "1"].append(row)

        for repeat, rows in sorted(rows_by_repeat.items(), key=lambda item: _numeric_key(item[0])):
            fidelity_name = _fidelity_name(source_csv, repeat)
            fidelity_dir = output / fidelity_name
            fidelity_dir.mkdir(parents=True, exist_ok=True)
            fidelity_id = _global_fidelity_id(source_csv, repeat, value_spaces)

            output_rows: list[dict[str, str]] = []
            for row in rows:
                normalized = _normalize_row(row, source_csv, repeat)
                row_id = normalized["ID"]
                config_id = row.get("config_id", "")

                client_log = client_logs.get((config_id, fidelity_id))
                if client_log is not None:
                    summary["linked_client_logs"] += 1
                else:
                    summary["missing_client_logs"] += 1

                server_log = server_logs.get(config_id)
                if server_log is not None:
                    summary["linked_server_logs"] += 1
                else:
                    summary["missing_server_logs"] += 1

                normalized["log-file"] = _write_combined_log_file(
                    client_log,
                    server_log,
                    fidelity_dir / "log_file" / f"log-{row_id}.txt",
                    fidelity_dir,
                )

                normalized["hw-file"] = ""
                summary["missing_hardware_files"] += 1

                output_rows.append(normalized)

            fieldnames = _fieldnames(source_csv.rows[0].keys())
            with (fidelity_dir / f"{fidelity_name}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(output_rows)

            summary["output_csv_files"] += 1
            summary["rows"] += len(output_rows)

    return summary


def _read_vllm_csvs(source: Path) -> list[VllmCsv]:
    csvs: list[VllmCsv] = []
    for csv_path in sorted(source.glob("*.csv"), key=_vllm_csv_sort_key):
        match = VLLM_CSV_RE.match(csv_path.name)
        if match is None:
            continue
        with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
        csvs.append(
            VllmCsv(
                path=csv_path,
                rate=match.group("rate"),
                burst=match.group("burst"),
                conc=match.group("conc"),
                prompts=match.group("prompts"),
                rows=rows,
            )
        )
    return csvs


def _index_client_logs(log_dir: Path) -> dict[tuple[str, int], Path]:
    indexed: dict[tuple[str, int], Path] = {}
    if not log_dir.is_dir():
        return indexed
    for path in sorted(log_dir.glob("client_config_*_fidelity_*.log")):
        match = CLIENT_LOG_RE.match(path.name)
        if match is None:
            continue
        key = (match.group("config_id"), int(match.group("fidelity_id")))
        indexed[key] = path
    return indexed


def _index_server_logs(log_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    if not log_dir.is_dir():
        return indexed
    for path in sorted(log_dir.glob("server_config_*.log")):
        match = SERVER_LOG_RE.match(path.name)
        if match is None:
            continue
        indexed[match.group("config_id")] = path
    return indexed


def _value_spaces(csv_inputs: list[VllmCsv]) -> dict[str, list[str]]:
    repeats = sorted(
        {
            row.get("repeat") or "1"
            for source_csv in csv_inputs
            for row in source_csv.rows
        },
        key=_numeric_key,
    )
    return {
        "prompts": sorted({item.prompts for item in csv_inputs}, key=_numeric_key),
        "conc": sorted({item.conc for item in csv_inputs}, key=_numeric_key),
        "rate": sorted({item.rate for item in csv_inputs}, key=_numeric_key),
        "burst": sorted({item.burst for item in csv_inputs}, key=_numeric_key),
        "repeat": repeats,
    }


def _global_fidelity_id(source_csv: VllmCsv, repeat: str, spaces: dict[str, list[str]]) -> int:
    prompt_index = spaces["prompts"].index(source_csv.prompts)
    conc_index = spaces["conc"].index(source_csv.conc)
    rate_index = spaces["rate"].index(source_csv.rate)
    burst_index = spaces["burst"].index(source_csv.burst)
    repeat_index = spaces["repeat"].index(repeat)
    return (
        (
            (
                (
                    prompt_index * len(spaces["conc"])
                    + conc_index
                )
                * len(spaces["rate"])
                + rate_index
            )
            * len(spaces["burst"])
            + burst_index
        )
        * len(spaces["repeat"])
        + repeat_index
        + 1
    )


def _normalize_row(row: dict[str, str], source_csv: VllmCsv, repeat: str) -> dict[str, str]:
    normalized = {
        "ID": row.get("id", ""),
    }
    for column, value in row.items():
        target = _normalize_column_name(column)
        if target is not None:
            normalized[target] = value
    return normalized


def _normalize_column_name(column: str) -> str | None:
    if column in {"id", "repeat"}:
        return None
    if column in AI_CONFIG_COLUMNS:
        return f"cfg-ai-{column}"
    if column in NON_AI_CONFIG_COLUMNS:
        return f"cfg-{column}"
    if column in OBJECTIVE_MAX_COLUMNS:
        return f"obj-{column}+"
    if column in OBJECTIVE_MIN_COLUMNS:
        return f"obj-{column}-"
    if column in COST_COLUMNS:
        return f"cost-{column}"
    return f"cfg-{column}"


def _fieldnames(source_columns: Iterable[str]) -> list[str]:
    fields = ["ID"]
    for column in source_columns:
        normalized = _normalize_column_name(column)
        if normalized is not None and normalized not in fields:
            fields.append(normalized)
    fields.extend(["hw-file", "log-file"])
    return fields


def _write_combined_log_file(
    client_log: Path | None,
    server_log: Path | None,
    destination: Path,
    base_dir: Path,
) -> str:
    sections = []
    if client_log is not None:
        sections.append(("CLIENT LOG", client_log.name, client_log.read_text(encoding="utf-8", errors="replace")))
    if server_log is not None:
        sections.append(("SERVER LOG", server_log.name, server_log.read_text(encoding="utf-8", errors="replace")))
    if not sections:
        return ""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for index, (title, source_name, content) in enumerate(sections):
            if index:
                handle.write("\n")
            handle.write(f"===== {title} =====\n")
            handle.write(f"source: {source_name}\n\n")
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
    return _relative_path(destination, base_dir)


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace(os.sep, "/")


def _fidelity_name(source_csv: VllmCsv, repeat: str) -> str:
    return f"{source_csv.rate}-{source_csv.burst}-{source_csv.conc}-{source_csv.prompts}-r{repeat}"


def _vllm_csv_sort_key(path: Path) -> tuple[float, float, int, int, str]:
    match = VLLM_CSV_RE.match(path.name)
    if match is None:
        return (float("inf"), float("inf"), 0, 0, path.name)
    return (
        float(match.group("rate")),
        float(match.group("burst")),
        int(match.group("conc")),
        int(match.group("prompts")),
        path.name,
    )


def _numeric_key(value: str) -> tuple[float, str]:
    try:
        return (float(value), value)
    except ValueError:
        return (float("inf"), value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize raw vLLM CSVs into LLMSYS-HPOBench format.")
    parser.add_argument("--source", default="vLLM", help="Raw vLLM directory.")
    parser.add_argument("--output", default="normalized/vLLM", help="Normalized output directory.")
    parser.add_argument(
        "--artifact-mode",
        choices=["reference", "copy", "hardlink"],
        default="reference",
        help="Legacy compatibility option. Combined log-file artifacts are always materialized.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output directory if it already exists.",
    )
    args = parser.parse_args()

    summary = normalize_vllm_dataset(
        args.source,
        args.output,
        artifact_mode=args.artifact_mode,
        overwrite=args.overwrite,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
