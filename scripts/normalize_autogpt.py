"""Normalize raw AutoGPT sampling results into the project data format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RAW_FIDELITY_RE = re.compile(
    r"^(?P<task_type>simple|moderate|complex|multi_stage)_r(?P<requests_count>\d+)_(?P<workload_category>.+)$"
)

AI_CONFIG_COLUMNS = [
    "model_name",
    "big_brain",
    "use_functions_api",
    "send_token_limit",
    "cognitive_strategy",
    "temperature",
    "max_tokens",
    "prompt_style",
]

NON_AI_CONFIG_COLUMNS = [
    "agent_config_id",
    "cycle_budget",
    "enabled_components",
    "allow_fs_access",
    "full_message_count",
    "shell_command_control",
    "task_max_duration_s",
    "information_availability",
    "num_runs",
]

OBJECTIVE_MAX_COLUMNS = [
    "success_rate",
    "num_success",
    "instruction_adherence_score",
    "correctness_score",
    "throughput_tasks_per_sec",
    "throughput_tokens_per_sec",
    "cycle_efficiency",
    "instruction_adherence_response_present",
    "instruction_adherence_no_error",
    "instruction_adherence_send_limit_respected",
    "instruction_adherence_finish_reason_ok",
]

OBJECTIVE_MIN_COLUMNS = [
    "error_rate",
    "timeout_rate",
    "step_timeout_rate",
    "wall_timeout_rate",
    "num_failures",
    "num_timeouts",
    "num_step_timeouts",
    "num_wall_timeouts",
    "avg_task_duration",
    "median_task_duration",
    "p95_task_duration",
    "p99_task_duration",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "avg_cycles_per_task",
    "avg_steps_per_wall_timeout",
    "retry_rate",
]

COST_COLUMNS = [
    "num_tasks",
    "total_duration",
    "total_cycles",
    "total_tokens",
    "avg_tokens_per_task",
    "tokens_per_request_avg",
    "tokens_per_request_std",
    "token_usage_prompt_tokens",
    "token_usage_completion_tokens",
    "token_usage_total_tokens",
    "estimated_cost_usd",
    "cost_total_cost_usd",
    "cost_prompt_cost_usd",
    "cost_completion_cost_usd",
]


@dataclass(frozen=True)
class RawFidelityDir:
    path: Path
    task_type: str
    requests_count: str
    workload_category: str

    @property
    def name(self) -> str:
        return f"{self.task_type}-req{self.requests_count}-{self.workload_category}"


def normalize_autogpt_dataset(
    root: str | Path = "experiment-data/Agent/autogpt",
    *,
    source_root: str | Path | None = None,
    remove_raw: bool = False,
    overwrite: bool = True,
) -> dict[str, int]:
    """Normalize AutoGPT ``large_scale`` samples in place."""

    system_root = Path(root).resolve()
    if not system_root.is_dir():
        raise FileNotFoundError(f"AutoGPT root not found: {system_root}")

    if source_root is not None:
        raw_root = Path(source_root).resolve()
    else:
        raw_root = system_root / "large_scale" if (system_root / "large_scale").is_dir() else system_root
    fidelities_root = raw_root / "fidelities"
    if not fidelities_root.is_dir():
        raise FileNotFoundError(f"AutoGPT raw fidelities directory not found: {fidelities_root}")

    raw_dirs = _discover_raw_fidelity_dirs(fidelities_root)
    if not raw_dirs:
        raise FileNotFoundError(f"No raw AutoGPT fidelity directories found under: {fidelities_root}")

    summary = {
        "input_fidelity_dirs": len(raw_dirs),
        "input_json_files": 0,
        "output_csv_files": 0,
        "rows": 0,
        "linked_log_files": 0,
        "missing_log_files": 0,
        "linked_hardware_files": 0,
        "missing_hardware_files": 0,
        "raw_root_removed": 0,
    }

    for raw_dir in raw_dirs:
        sample_paths = sorted(
            path for path in raw_dir.path.glob("*.json") if not path.name.startswith("_")
        )
        if not sample_paths:
            continue

        output_dir = system_root / raw_dir.name
        if output_dir.exists() and overwrite:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for row_id, sample_path in enumerate(sample_paths, start=1):
            with sample_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                payload = json.load(handle)
            row = _normalize_row(row_id, payload)

            hw_ref = _write_hardware_file(payload.get("hardware"), output_dir, row_id)
            if hw_ref:
                summary["linked_hardware_files"] += 1
            else:
                summary["missing_hardware_files"] += 1
            row["hw-file"] = hw_ref

            log_ref = _write_log_file(
                sample_path.with_suffix(".log"),
                payload.get("task_results"),
                payload.get("server_log_offsets"),
                raw_root,
                output_dir,
                row_id,
            )
            if log_ref:
                summary["linked_log_files"] += 1
            else:
                summary["missing_log_files"] += 1
            row["log-file"] = log_ref

            rows.append(row)

        csv_path = output_dir / f"{raw_dir.name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        summary["input_json_files"] += len(sample_paths)
        summary["output_csv_files"] += 1
        summary["rows"] += len(rows)

    if remove_raw and source_root is None and raw_root != system_root and raw_root.is_dir():
        shutil.rmtree(raw_root)
        summary["raw_root_removed"] = 1

    return summary


def _discover_raw_fidelity_dirs(fidelities_root: Path) -> list[RawFidelityDir]:
    dirs: list[RawFidelityDir] = []
    for path in sorted(item for item in fidelities_root.iterdir() if item.is_dir()):
        match = RAW_FIDELITY_RE.match(path.name)
        if match is None:
            continue
        dirs.append(
            RawFidelityDir(
                path=path,
                task_type=match.group("task_type"),
                requests_count=match.group("requests_count"),
                workload_category=match.group("workload_category"),
            )
        )
    return dirs


def _normalize_row(row_id: int, payload: dict[str, Any]) -> dict[str, str]:
    fidelity_config = payload.get("fidelity_config") or {}
    agent_config = payload.get("agent_config") or {}
    execution_constraints = agent_config.get("execution_constraints") or {}
    metrics = _flatten_metrics(payload.get("metrics") or {})

    config_values = {
        "agent_config_id": payload.get("agent_config_id", ""),
        "cycle_budget": fidelity_config.get("cycle_budget", ""),
        "enabled_components": _join_list(fidelity_config.get("enabled_components")),
        "allow_fs_access": fidelity_config.get("allow_fs_access", ""),
        "full_message_count": fidelity_config.get("full_message_count", ""),
        "shell_command_control": fidelity_config.get("shell_command_control", ""),
        "task_max_duration_s": execution_constraints.get("task_max_duration_s", ""),
        "information_availability": fidelity_config.get("information_availability", ""),
        "num_runs": payload.get("num_runs", ""),
    }

    row = {"ID": str(row_id)}
    for column in NON_AI_CONFIG_COLUMNS:
        row[f"cfg-{column}"] = _format_value(config_values.get(column, ""))
    for column in AI_CONFIG_COLUMNS:
        row[f"cfg-ai-{column}"] = _format_value(fidelity_config.get(column, ""))
    for column in COST_COLUMNS:
        row[f"cost-{column}"] = _format_value(metrics.get(column, ""))
    for column in OBJECTIVE_MAX_COLUMNS:
        row[f"obj-{column}+"] = _format_value(metrics.get(column, ""))
    for column in OBJECTIVE_MIN_COLUMNS:
        row[f"obj-{column}-"] = _format_value(metrics.get(column, ""))
    return row


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(metrics)
    token_usage = metrics.get("token_usage_breakdown") or {}
    flattened["token_usage_prompt_tokens"] = token_usage.get("prompt_tokens", "")
    flattened["token_usage_completion_tokens"] = token_usage.get("completion_tokens", "")
    flattened["token_usage_total_tokens"] = token_usage.get("total_tokens", "")

    cost_breakdown = metrics.get("cost_breakdown") or {}
    flattened["cost_total_cost_usd"] = cost_breakdown.get("total_cost_usd", "")
    flattened["cost_prompt_cost_usd"] = cost_breakdown.get("prompt_cost_usd", "")
    flattened["cost_completion_cost_usd"] = cost_breakdown.get("completion_cost_usd", "")

    latency = metrics.get("latency_percentiles") or {}
    flattened["latency_p50"] = latency.get("p50", "")
    flattened["latency_p90"] = latency.get("p90", "")
    flattened["latency_p95"] = latency.get("p95", "")
    flattened["latency_p99"] = latency.get("p99", "")

    adherence = metrics.get("instruction_adherence_breakdown") or {}
    flattened["instruction_adherence_response_present"] = adherence.get("response_present", "")
    flattened["instruction_adherence_no_error"] = adherence.get("no_error", "")
    flattened["instruction_adherence_send_limit_respected"] = adherence.get("send_limit_respected", "")
    flattened["instruction_adherence_finish_reason_ok"] = adherence.get("finish_reason_ok", "")
    return flattened


def _write_hardware_file(hardware: Any, output_dir: Path, row_id: int) -> str:
    if not hardware:
        return ""
    destination = output_dir / "hw_file" / f"hw-{row_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(hardware, indent=2, sort_keys=True), encoding="utf-8")
    return _relative_path(destination, output_dir)


def _write_log_file(
    sample_log: Path,
    task_results: Any,
    server_log_offsets: Any,
    raw_root: Path,
    output_dir: Path,
    row_id: int,
) -> str:
    sections: list[tuple[str, str]] = []
    if sample_log.is_file():
        content = sample_log.read_text(encoding="utf-8", errors="replace")
        sections.append(("SAMPLE LOG", f"source: {sample_log.name}\n\n{content}"))
    task_results_text = _format_task_results(task_results)
    if task_results_text:
        sections.append(("TASK RESULTS", task_results_text))
    sections.extend(_server_log_sections(server_log_offsets, raw_root))
    if server_log_offsets:
        content = json.dumps(server_log_offsets, indent=2, sort_keys=True)
        sections.append(("SERVER LOG OFFSETS", content))
    if not sections:
        return ""

    destination = output_dir / "log_file" / f"log-{row_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for index, (title, content) in enumerate(sections):
            if index:
                handle.write("\n")
            handle.write(f"===== {title} =====\n")
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
    return _relative_path(destination, output_dir)


def _server_log_sections(server_log_offsets: Any, raw_root: Path) -> list[tuple[str, str]]:
    if not isinstance(server_log_offsets, dict):
        return []

    sections: list[tuple[str, str]] = []
    for name in ("autogpt", "vllm"):
        offsets = server_log_offsets.get(name)
        if not isinstance(offsets, dict):
            continue
        log_path = _resolve_server_log_path(name, raw_root, offsets.get("path"))
        content = _read_log_slice(log_path, offsets)
        if not content:
            continue
        title = f"{name.upper()} SERVER LOG"
        source = offsets.get("path") or log_path.name
        sections.append((title, f"source: {source}\nlocal_source: {_relative_path(log_path, raw_root.parent)}\n\n{content}"))
    return sections


def _resolve_server_log_path(name: str, raw_root: Path, source_path: Any) -> Path:
    candidates: list[Path] = []
    if source_path:
        candidates.append(raw_root.parent / Path(str(source_path)).name)
    candidates.append(raw_root.parent / ("autogpt_server.log" if name == "autogpt" else f"{name}.log"))
    candidates.append(raw_root / ("autogpt_server.log" if name == "autogpt" else f"{name}.log"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _read_log_slice(log_path: Path, offsets: dict[str, Any]) -> str:
    if not log_path.is_file():
        return ""
    try:
        start = int(offsets.get("start_byte", 0))
        end = int(offsets.get("end_byte", 0))
    except (TypeError, ValueError):
        return ""
    if start < 0 or end <= start:
        return ""
    with log_path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    return data.decode("utf-8", errors="replace")


def _format_task_results(task_results: Any) -> str:
    if not task_results:
        return ""
    if not isinstance(task_results, list):
        return json.dumps(task_results, indent=2, sort_keys=True)

    sections = []
    for index, result in enumerate(task_results, start=1):
        if not isinstance(result, dict):
            sections.append(f"--- TASK {index} ---\n{_format_value(result)}")
            continue
        lines = [f"--- TASK {index} ---"]
        for key in ("task_id", "run_id", "status", "duration", "error"):
            if key in result and result.get(key) not in (None, ""):
                lines.append(f"{key}: {_format_value(result.get(key))}")
        output = result.get("output")
        if output not in (None, ""):
            lines.append("output:")
            lines.append(_format_value(output))
        metadata = result.get("metadata")
        if metadata:
            lines.append("metadata:")
            lines.append(json.dumps(metadata, indent=2, sort_keys=True))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _fieldnames() -> list[str]:
    fields = ["ID"]
    fields.extend(f"cfg-{column}" for column in NON_AI_CONFIG_COLUMNS)
    fields.extend(f"cfg-ai-{column}" for column in AI_CONFIG_COLUMNS)
    fields.extend(f"cost-{column}" for column in COST_COLUMNS)
    fields.extend(f"obj-{column}+" for column in OBJECTIVE_MAX_COLUMNS)
    fields.extend(f"obj-{column}-" for column in OBJECTIVE_MIN_COLUMNS)
    fields.extend(["hw-file", "log-file"])
    return fields


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return "+".join(str(item) for item in value)
    return _format_value(value)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace(os.sep, "/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize raw AutoGPT large-scale samples in place.")
    parser.add_argument("--root", default="experiment-data/Agent/autogpt", help="AutoGPT system root.")
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Optional raw AutoGPT large_scale root. Use this when raw files live outside "
            "the normalized system root, for example experiment-data/autogpt_original/large_scale."
        ),
    )
    parser.add_argument(
        "--remove-raw",
        action="store_true",
        help="Remove the raw large_scale directory after normalized artifacts are written.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not delete an existing normalized fidelity directory before rewriting it.",
    )
    args = parser.parse_args()

    summary = normalize_autogpt_dataset(
        args.root,
        source_root=args.source_root,
        remove_raw=args.remove_raw,
        overwrite=not args.no_overwrite,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
