"""Normalize raw SGLang JSON sampling results into the project data format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


RAW_FIDELITY_RE = re.compile(
    r"^rate(?P<rate>[^_]+)"
    r"(?:_burst(?P<burst>[^_]+))?"
    r"_conc(?P<conc>[^_]+)"
    r"_groups(?P<groups>[^_]+)"
    r"_ppg(?P<ppg>[^_]+)"
    r"_syslen(?P<syslen>[^_]+)"
    r"_qlen(?P<qlen>[^_]+)"
    r"_olen(?P<olen>[^_]+)$"
)
CLIENT_LOG_RE = re.compile(
    r"^client_config_(?P<config_id>[^_]+)_fidelity_(?P<fidelity_id>[^_]+)_(?P<stamp>\d{8}_\d{6})\.log$"
)
SERVER_LOG_RE = re.compile(r"^server_config_(?P<config_id>[^_]+)_(?P<stamp>\d{8}_\d{6})\.log$")
SERVER_TIMESTAMP_RE = re.compile(r"\[(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
CLIENT_STARTED_RE = re.compile(r"Started at:\s*(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
CLIENT_COMPLETED_RE = re.compile(r"Completed at:\s*(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

AI_CONFIG_COLUMNS = [
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "frequency_penalty",
]

NON_AI_CONFIG_COLUMNS = [
    "config_id",
    "tp_size",
    "pp_size",
    "max_running_requests",
    "max_total_tokens",
    "chunked_prefill_size",
    "gpu_memory_utilization",
    "attention_backend",
    "context_length",
    "enable_torch_compile",
    "enable_p2p_check",
    "disable_radix_cache",
]

OBJECTIVE_MAX_COLUMNS = [
    "completed",
    "request_throughput",
    "input_throughput",
    "output_throughput",
]

OBJECTIVE_MIN_COLUMNS = [
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "std_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
]

COST_COLUMNS = [
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "total_output_tokens_retokenized",
    "concurrency",
]

FAILED_CONTEXT_LIMIT = 8192
FAILED_LOG_CSV = "sglang_multi_fidelity_benchmark_log.csv"


@dataclass(frozen=True)
class RawFidelityDir:
    path: Path
    request_rate: str
    burstiness: str
    max_concurrency: str
    gsp_num_groups: str
    gsp_prompts_per_group: str
    gsp_system_prompt_len: str
    gsp_question_len: str
    gsp_output_len: str

    @property
    def name(self) -> str:
        return (
            f"{self.request_rate}-{self.burstiness}-{self.max_concurrency}-"
            f"{self.gsp_num_groups}-{self.gsp_system_prompt_len}"
        )


@dataclass(frozen=True)
class ResultFile:
    path: Path
    config_id: str
    fidelity_id: str
    timestamp: str
    file_config: dict[str, Any]


def normalize_sglang_dataset(
    root: str | Path = "experiment-data/Engine/SGLang",
    *,
    remove_raw: bool = False,
    overwrite: bool = True,
    padding_seconds: int = 3,
) -> dict[str, int]:
    """Normalize raw SGLang JSON samples in place.

    Output fidelity names follow the order in ``experiment-data/tab-format.tex``:
    ``request_rate-burstiness-max_concurrency-gsp_num_groups-gsp_system_prompt_len``.
    Fixed generated-shared-prefix axes such as prompts per group, question
    length, and output length are not duplicated in the fidelity file name.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"SGLang root not found: {root_path}")

    raw_dirs = _discover_raw_fidelity_dirs(root_path)
    if not raw_dirs:
        raise FileNotFoundError(f"No raw SGLang fidelity directories found under: {root_path}")

    client_logs = _index_client_logs(root_path / "logs")
    server_logs = _index_server_logs(root_path / "logs")
    summary = {
        "input_fidelity_dirs": len(raw_dirs),
        "input_json_files": 0,
        "output_csv_files": 0,
        "rows": 0,
        "linked_client_logs": 0,
        "missing_client_logs": 0,
        "linked_server_logs": 0,
        "missing_server_logs": 0,
        "linked_hardware_files": 0,
        "missing_hardware_files": 0,
        "raw_dirs_removed": 0,
    }

    for raw_dir in raw_dirs:
        result_files = _discover_result_files(raw_dir.path)
        if not result_files:
            continue

        fidelity_dir = root_path / raw_dir.name
        if fidelity_dir.exists() and overwrite:
            shutil.rmtree(fidelity_dir)
        fidelity_dir.mkdir(parents=True, exist_ok=True)

        output_rows: list[dict[str, Any]] = []
        for index, result_file in enumerate(result_files, start=1):
            with result_file.path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                result_payload = json.load(handle)

            client_log = _select_latest(client_logs.get((result_file.config_id, result_file.fidelity_id), []))
            server_log = _select_latest(server_logs.get(result_file.config_id, []))
            if client_log is not None:
                summary["linked_client_logs"] += 1
            else:
                summary["missing_client_logs"] += 1
            if server_log is not None:
                summary["linked_server_logs"] += 1
            else:
                summary["missing_server_logs"] += 1

            row = _normalize_row(
                row_id=index,
                result_payload=result_payload,
                result_file=result_file,
                client_log=client_log,
            )
            row["log-file"] = _write_combined_log_file(
                client_log,
                server_log,
                destination=fidelity_dir / "log_file" / f"log-{index}.txt",
                base_dir=fidelity_dir,
                padding_seconds=padding_seconds,
            )
            row["hw-file"] = ""
            summary["missing_hardware_files"] += 1
            output_rows.append(row)

        csv_path = fidelity_dir / f"{raw_dir.name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output_rows)

        summary["input_json_files"] += len(result_files)
        summary["output_csv_files"] += 1
        summary["rows"] += len(output_rows)

    if remove_raw:
        for raw_dir in raw_dirs:
            if raw_dir.path.exists():
                shutil.rmtree(raw_dir.path)
                summary["raw_dirs_removed"] += 1
        logs_dir = root_path / "logs"
        if logs_dir.is_dir():
            shutil.rmtree(logs_dir)

    return summary


def include_sglang_failed_log_samples(
    root: str | Path = "experiment-data/Engine/SGLang",
    *,
    log_csv_name: str = FAILED_LOG_CSV,
    exit_code: int = -3,
) -> dict[str, int]:
    """Append known failed SGLang runs from the multi-fidelity log as valid samples.

    ``exit_code=-3`` records runs that were skipped because the sampled
    ``context_length`` exceeded the active SGLang limit. They are valid
    configuration observations: the objective values are unavailable except
    completed=0, and the reason is materialized as the sample log.
    """

    root_path = Path(root).resolve()
    log_csv = root_path / log_csv_name
    if not log_csv.is_file():
        log_csv = root_path / "log_file" / log_csv_name
    if not log_csv.is_file():
        log_csv = root_path / "raw_metadata" / log_csv_name
    if not log_csv.is_file():
        raise FileNotFoundError(f"SGLang multi-fidelity log CSV not found: {log_csv}")

    summary = {
        "failed_log_rows": 0,
        "appended_failed_rows": 0,
        "skipped_existing_failed_rows": 0,
        "created_fidelity_csv_files": 0,
    }

    pending_by_csv: dict[Path, list[dict[str, Any]]] = {}
    with log_csv.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for source_line, raw_row in enumerate(reader, start=2):
            if _stringify(raw_row.get("exit_code")) != str(exit_code):
                continue

            summary["failed_log_rows"] += 1
            fidelity_name = _failed_log_fidelity_name(raw_row)
            fidelity_dir = root_path / fidelity_name
            csv_path = fidelity_dir / f"{fidelity_name}.csv"
            if _csv_has_config_id(csv_path, f"failed-{source_line}"):
                summary["skipped_existing_failed_rows"] += 1
                continue
            log_file = _failed_log_file_name(csv_path, pending_count=len(pending_by_csv.get(csv_path, [])))

            if not csv_path.exists():
                summary["created_fidelity_csv_files"] += 1

            row = _normalize_failed_log_row(raw_row, source_line=source_line, log_file=log_file)
            _write_failed_sample_log(
                raw_row,
                source_line=source_line,
                exit_code=exit_code,
                destination=fidelity_dir / log_file,
                source_name=log_csv.name,
            )
            pending_by_csv.setdefault(csv_path, []).append(row)

    for csv_path, rows in pending_by_csv.items():
        _append_rows_to_csv(csv_path, rows)
        summary["appended_failed_rows"] += len(rows)

    return summary


def _discover_raw_fidelity_dirs(root: Path) -> list[RawFidelityDir]:
    dirs: list[RawFidelityDir] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        match = RAW_FIDELITY_RE.match(path.name)
        if match is None:
            continue
        dirs.append(
            RawFidelityDir(
                path=path,
                request_rate=match.group("rate"),
                burstiness=match.group("burst") or "1.0",
                max_concurrency=match.group("conc"),
                gsp_num_groups=match.group("groups"),
                gsp_prompts_per_group=match.group("ppg"),
                gsp_system_prompt_len=match.group("syslen"),
                gsp_question_len=match.group("qlen"),
                gsp_output_len=match.group("olen"),
            )
        )
    return dirs


def _failed_log_fidelity_name(row: dict[str, Any]) -> str:
    return (
        f"{_stringify(row.get('request_rate'))}-"
        f"{_stringify(row.get('burstiness'))}-"
        f"{_stringify(row.get('max_concurrency'))}-"
        f"{_stringify(row.get('gsp_num_groups'))}-"
        f"{_stringify(row.get('gsp_system_prompt_len'))}"
    )


def _failed_log_file_name(csv_path: Path, *, pending_count: int = 0) -> str:
    next_id = _next_csv_row_id(csv_path) + pending_count
    return f"log_file/log-{next_id}.txt"


def _next_csv_row_id(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 1
    with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        ids = []
        for row in reader:
            try:
                ids.append(int(_stringify(row.get("ID"))))
            except ValueError:
                continue
    return (max(ids) if ids else 0) + 1


def _csv_has_config_id(csv_path: Path, config_id: str) -> bool:
    if not csv_path.is_file():
        return False
    with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        return any(row.get("cfg-config_id") == config_id for row in reader)


def _normalize_failed_log_row(raw_row: dict[str, Any], *, source_line: int, log_file: str) -> dict[str, Any]:
    extra_body = _parse_extra_body(raw_row.get("extra_body"))
    row: dict[str, Any] = {"ID": ""}

    for column in NON_AI_CONFIG_COLUMNS:
        if column == "config_id":
            value = f"failed-{source_line}"
        else:
            value = raw_row.get(column, "")
        row[f"cfg-{column}"] = _format_value(value)

    for column in AI_CONFIG_COLUMNS:
        row[f"cfg-ai-{column}"] = _format_value(extra_body.get(column, ""))

    for column in COST_COLUMNS:
        if column == "concurrency":
            value = ""
        elif column == "total_output_tokens_retokenized":
            value = raw_row.get("total_output_tokens", "")
        else:
            value = raw_row.get(column, "")
        row[f"cost-{column}"] = _format_value(value)

    for column in OBJECTIVE_MAX_COLUMNS:
        value = raw_row.get(column, "")
        if column == "completed":
            value = raw_row.get("completed", "0") or "0"
        row[f"obj-{column}+"] = _format_value(value)

    for column in OBJECTIVE_MIN_COLUMNS:
        row[f"obj-{column}-"] = _format_value(raw_row.get(column, ""))

    row["hw-file"] = ""
    row["log-file"] = log_file
    return row


def _write_failed_sample_log(
    raw_row: dict[str, Any],
    *,
    source_line: int,
    exit_code: int,
    destination: Path,
    source_name: str,
) -> None:
    context_length = _stringify(raw_row.get("context_length"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_payload = json.dumps(raw_row, ensure_ascii=False, indent=2)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write("===== ERROR LOG =====\n")
        handle.write(f"source: {source_name}:{source_line}\n")
        handle.write(f"exit_code: {exit_code}\n")
        handle.write(
            f"error: Skip config {source_line}: context_length={context_length} "
            f"exceeds {FAILED_CONTEXT_LIMIT}\n"
        )
        handle.write(f"Result saved: exit_code={exit_code}\n\n")
        handle.write("===== RAW LOG ROW =====\n")
        handle.write(raw_payload)
        handle.write("\n")


def _append_rows_to_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    fieldnames = _fieldnames()
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)

    next_id = _next_row_id(existing_rows)
    for row in rows:
        row["ID"] = str(next_id)
        next_id += 1
        existing_rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)


def _next_row_id(rows: list[dict[str, Any]]) -> int:
    ids: list[int] = []
    for row in rows:
        try:
            ids.append(int(row.get("ID", "")))
        except (TypeError, ValueError):
            continue
    return max(ids, default=0) + 1


def _parse_extra_body(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _discover_result_files(raw_dir: Path) -> list[ResultFile]:
    results = []
    for path in sorted(raw_dir.glob("*.json"), key=lambda item: _result_sort_key(item.name)):
        parsed = _parse_result_filename(path.name)
        if parsed is None:
            continue
        results.append(ResultFile(path=path, **parsed))
    return results


def _parse_result_filename(name: str) -> dict[str, Any] | None:
    if not name.endswith(".json"):
        return None
    tokens = name[:-5].split("_")
    if len(tokens) < 8:
        return None

    config: dict[str, Any] = {}
    config_id = ""
    fidelity_id = ""
    timestamp_parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("tp"):
            config["tp_size"] = token[2:]
        elif token.startswith("pp"):
            config["pp_size"] = token[2:]
        elif token.startswith("reqs"):
            config["max_running_requests"] = token[4:]
        elif token.startswith("tokens"):
            config["max_total_tokens"] = token[6:]
        elif token.startswith("chunk"):
            config["chunked_prefill_size"] = token[5:]
        elif token.startswith("attn"):
            attention = token[4:]
            if index + 1 < len(tokens) and tokens[index + 1] == "native":
                attention = f"{attention}_native"
                index += 1
            config["attention_backend"] = attention
        elif token.startswith("ctx"):
            config["context_length"] = token[3:]
        elif token.startswith("mem"):
            config["gpu_memory_utilization"] = _percent_to_fraction(token[3:])
        elif token.startswith("config"):
            config_id = token[6:]
        elif token.startswith("fidelity"):
            fidelity_id = token[8:]
        elif re.fullmatch(r"\d{8}", token) and index + 1 < len(tokens):
            timestamp_parts = [token, tokens[index + 1]]
            index += 1
        index += 1

    if not config_id or not fidelity_id or len(timestamp_parts) != 2:
        return None
    config["config_id"] = config_id
    return {
        "config_id": config_id,
        "fidelity_id": fidelity_id,
        "timestamp": "_".join(timestamp_parts),
        "file_config": config,
    }


def _index_client_logs(log_dir: Path) -> dict[tuple[str, str], list[Path]]:
    indexed: dict[tuple[str, str], list[Path]] = {}
    if not log_dir.is_dir():
        return indexed
    for path in sorted(log_dir.glob("client_config_*_fidelity_*.log")):
        match = CLIENT_LOG_RE.match(path.name)
        if match is None:
            continue
        key = (match.group("config_id"), match.group("fidelity_id"))
        indexed.setdefault(key, []).append(path)
    return indexed


def _index_server_logs(log_dir: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    if not log_dir.is_dir():
        return indexed
    for path in sorted(log_dir.glob("server_config_*.log")):
        match = SERVER_LOG_RE.match(path.name)
        if match is None:
            continue
        indexed.setdefault(match.group("config_id"), []).append(path)
    return indexed


def _select_latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths)[-1]


def _normalize_row(
    *,
    row_id: int,
    result_payload: dict[str, Any],
    result_file: ResultFile,
    client_log: Path | None,
) -> dict[str, Any]:
    client_config: dict[str, Any] = {}
    sampling_params: dict[str, Any] = {}
    if client_log is not None:
        text = client_log.read_text(encoding="utf-8", errors="replace")
        client_config = _extract_json_block(text, "SGLang Config") or {}
        sampling_params = _extract_json_block(text, "Sampling Params") or {}
        extra_body = client_config.get("extra_body")
        if isinstance(extra_body, str) and not sampling_params:
            try:
                sampling_params = json.loads(extra_body)
            except json.JSONDecodeError:
                sampling_params = {}

    merged_config = dict(result_file.file_config)
    merged_config.update({key: value for key, value in client_config.items() if key in NON_AI_CONFIG_COLUMNS})
    merged_config["config_id"] = result_file.config_id

    row: dict[str, Any] = {"ID": str(row_id)}
    for column in NON_AI_CONFIG_COLUMNS:
        row[f"cfg-{column}"] = _format_value(merged_config.get(column, ""))
    for column in AI_CONFIG_COLUMNS:
        row[f"cfg-ai-{column}"] = _format_value(sampling_params.get(column, ""))
    for column in COST_COLUMNS:
        row[f"cost-{column}"] = _format_value(result_payload.get(column, ""))
    for column in OBJECTIVE_MAX_COLUMNS:
        row[f"obj-{column}+"] = _format_value(result_payload.get(column, ""))
    for column in OBJECTIVE_MIN_COLUMNS:
        row[f"obj-{column}-"] = _format_value(result_payload.get(column, ""))
    return row


def _extract_json_block(text: str, title: str) -> dict[str, Any] | None:
    marker = f"{title}:"
    start = text.find(marker)
    if start < 0:
        return None
    brace_start = text.find("{", start)
    if brace_start < 0:
        return None

    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                block = text[brace_start : index + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    return None
    return None


def _write_combined_log_file(
    client_log: Path | None,
    server_log: Path | None,
    *,
    destination: Path,
    base_dir: Path,
    padding_seconds: int,
) -> str:
    sections: list[tuple[str, str, str]] = []
    client_window: tuple[datetime, datetime] | None = None

    if client_log is not None:
        client_content = client_log.read_text(encoding="utf-8", errors="replace")
        sections.append(("CLIENT LOG", client_log.name, client_content))
        client_window = _client_window(client_content)

    if server_log is not None:
        server_content = server_log.read_text(encoding="utf-8", errors="replace")
        sliced_lines, selected_count = _slice_server_log(server_content, client_window, padding_seconds)
        server_header = [
            f"source: {server_log.name}",
            f"client_window: {_format_window(client_window)}",
            f"padding_seconds: {padding_seconds}",
            f"selected_lines: {selected_count}",
            "",
        ]
        sections.append(("SERVER LOG", "", "\n".join(server_header + sliced_lines)))

    if not sections:
        return ""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for index, (title, source_name, content) in enumerate(sections):
            if index:
                handle.write("\n")
            handle.write(f"===== {title} =====\n")
            if source_name:
                handle.write(f"source: {source_name}\n\n")
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
    return os.path.relpath(destination.resolve(), base_dir.resolve()).replace(os.sep, "/")


def _client_window(client_content: str) -> tuple[datetime, datetime] | None:
    started = CLIENT_STARTED_RE.search(client_content)
    completed = CLIENT_COMPLETED_RE.search(client_content)
    if started is None or completed is None:
        return None
    try:
        return (
            datetime.strptime(started.group("stamp"), "%Y-%m-%d %H:%M:%S"),
            datetime.strptime(completed.group("stamp"), "%Y-%m-%d %H:%M:%S"),
        )
    except ValueError:
        return None


def _slice_server_log(
    server_content: str,
    client_window: tuple[datetime, datetime] | None,
    padding_seconds: int,
) -> tuple[list[str], int]:
    if client_window is None:
        return ([], 0)

    start = client_window[0] - timedelta(seconds=padding_seconds)
    end = client_window[1] + timedelta(seconds=padding_seconds)
    selected: list[str] = []
    active = False
    for line in server_content.splitlines():
        match = SERVER_TIMESTAMP_RE.search(line)
        if match is not None:
            try:
                stamp = datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                stamp = None
            active = stamp is not None and start <= stamp <= end
        if active:
            selected.append(line)
    return (selected, len(selected))


def _format_window(client_window: tuple[datetime, datetime] | None) -> str:
    if client_window is None:
        return ""
    return f"{client_window[0].isoformat(sep=' ')}..{client_window[1].isoformat(sep=' ')}"


def _fieldnames() -> list[str]:
    fields = ["ID"]
    fields.extend(f"cfg-{column}" for column in NON_AI_CONFIG_COLUMNS)
    fields.extend(f"cfg-ai-{column}" for column in AI_CONFIG_COLUMNS)
    fields.extend(f"cost-{column}" for column in COST_COLUMNS)
    fields.extend(f"obj-{column}+" for column in OBJECTIVE_MAX_COLUMNS)
    fields.extend(f"obj-{column}-" for column in OBJECTIVE_MIN_COLUMNS)
    fields.extend(["hw-file", "log-file"])
    return fields


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _percent_to_fraction(value: str) -> str:
    try:
        return str(float(value) / 100)
    except ValueError:
        return value


def _result_sort_key(name: str) -> tuple[int, int, str]:
    parsed = _parse_result_filename(name)
    if parsed is None:
        return (10**12, 10**12, name)
    return (int(parsed["config_id"]), int(parsed["fidelity_id"]), parsed["timestamp"])


def _numeric_key(value: str) -> tuple[float, str]:
    if value == "inf":
        return (float("inf"), value)
    try:
        return (float(value), value)
    except ValueError:
        return (float("inf"), value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize raw SGLang JSON samples in place.")
    parser.add_argument("--root", default="experiment-data/Engine/SGLang", help="Raw SGLang system root.")
    parser.add_argument(
        "--remove-raw",
        action="store_true",
        help="Remove raw JSON fidelity directories and top-level logs after normalized artifacts are written.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not delete an existing normalized fidelity directory before rewriting it.",
    )
    parser.add_argument("--padding-seconds", type=int, default=3, help="Server-log slicing padding around client window.")
    args = parser.parse_args()

    summary = normalize_sglang_dataset(
        args.root,
        remove_raw=args.remove_raw,
        overwrite=not args.no_overwrite,
        padding_seconds=args.padding_seconds,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
