"""Slice vLLM server logs to the time window of each sampled client run."""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable


TIME_RE = re.compile(r"^\[(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\]")
CLIENT_START_MARKER = "Starting vLLM Sampler Benchmark"
CLIENT_END_MARKERS = (
    "=== Client finished with exit code:",
    "Benchmark completed successfully",
)


@dataclass(frozen=True)
class _SliceTask:
    client_path: Path
    server_path: Path
    start: int
    end: int


def extract_client_window(client_log: str | Path) -> tuple[int, int]:
    """Return start/end seconds-of-day for one client benchmark log."""

    path = Path(client_log)
    start: int | None = None
    end: int | None = None
    last_timestamp: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = _timestamp_seconds(line)
            if timestamp is not None:
                last_timestamp = timestamp
            if CLIENT_START_MARKER in line and timestamp is not None and start is None:
                start = timestamp
            if any(marker in line for marker in CLIENT_END_MARKERS) and timestamp is not None:
                end = timestamp

    if start is None:
        raise ValueError(f"Could not find client start marker in {path}")
    if end is None:
        end = last_timestamp
    if end is None:
        raise ValueError(f"Could not find client end timestamp in {path}")
    return start, end


def slice_vllm_server_logs(
    root: str | Path,
    *,
    padding_seconds: int = 3,
    backup: bool = False,
) -> dict[str, int]:
    """Slice all server logs referenced by normalized vLLM CSVs under ``root``."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"vLLM root not found: {root_path}")

    summary = {
        "csvs_scanned": 0,
        "rows_scanned": 0,
        "server_logs_sliced": 0,
        "server_logs_parsed": 0,
        "missing_client_logs": 0,
        "missing_server_logs": 0,
        "missing_server_references": 0,
        "window_parse_failures": 0,
        "empty_slices": 0,
        "already_sliced": 0,
        "combined_logs_sliced": 0,
        "missing_log_files": 0,
    }

    tasks_by_server: dict[tuple[int, int], list[_SliceTask]] = {}
    server_paths_by_key: dict[tuple[int, int], Path] = {}

    for csv_path in _main_csvs(root_path):
        summary["csvs_scanned"] += 1
        fidelity_dir = csv_path.parent
        with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                summary["rows_scanned"] += 1
                combined_ref = (row.get("log-file") or "").strip()
                if combined_ref:
                    combined_path = fidelity_dir / combined_ref
                    if not combined_path.is_file():
                        summary["missing_log_files"] += 1
                        continue
                    if _looks_combined_already_sliced(combined_path):
                        summary["already_sliced"] += 1
                        continue
                    result = _slice_combined_log_file(
                        combined_path,
                        padding_seconds=padding_seconds,
                        backup=backup,
                    )
                    if result == "sliced":
                        summary["server_logs_sliced"] += 1
                        summary["combined_logs_sliced"] += 1
                    elif result == "empty":
                        summary["server_logs_sliced"] += 1
                        summary["combined_logs_sliced"] += 1
                        summary["empty_slices"] += 1
                    elif result == "window_parse_failed":
                        summary["window_parse_failures"] += 1
                    elif result == "missing_server":
                        summary["missing_server_references"] += 1
                    continue

                client_ref = (row.get("log-client-file") or "").strip()
                server_ref = (row.get("log-server-file") or "").strip()
                if not server_ref:
                    summary["missing_server_references"] += 1
                    continue
                if not client_ref:
                    summary["missing_client_logs"] += 1
                    continue

                client_path = fidelity_dir / client_ref
                server_path = fidelity_dir / server_ref
                if not client_path.is_file():
                    summary["missing_client_logs"] += 1
                    continue
                if not server_path.is_file():
                    summary["missing_server_logs"] += 1
                    continue
                if _looks_already_sliced(server_path):
                    summary["already_sliced"] += 1
                    continue

                try:
                    start, end = extract_client_window(client_path)
                except ValueError:
                    summary["window_parse_failures"] += 1
                    continue

                stat = server_path.stat()
                key = (stat.st_dev, stat.st_ino)
                server_paths_by_key.setdefault(key, server_path)
                tasks_by_server.setdefault(key, []).append(
                    _SliceTask(
                        client_path=client_path,
                        server_path=server_path,
                        start=start,
                        end=end,
                    )
                )

    for key, tasks in tasks_by_server.items():
        timed_lines = _parse_timed_lines(server_paths_by_key[key])
        summary["server_logs_parsed"] += 1
        timestamps = [timestamp for timestamp, _line in timed_lines]
        for task in tasks:
            aligned_start, aligned_end = _align_window_to_timeline(
                task.start,
                task.end,
                timestamps,
            )
            sliced_lines = _select_lines(
                timed_lines,
                timestamps,
                aligned_start - padding_seconds,
                aligned_end + padding_seconds,
            )
            if not sliced_lines:
                summary["empty_slices"] += 1

            if backup:
                backup_path = task.server_path.with_suffix(task.server_path.suffix + ".bak")
                if not backup_path.exists():
                    shutil.copy2(task.server_path, backup_path)

            _write_sliced_log(
                task.server_path,
                task.client_path,
                task.start,
                task.end,
                padding_seconds,
                sliced_lines,
            )
            summary["server_logs_sliced"] += 1

    return summary


def _main_csvs(root: Path) -> list[Path]:
    return sorted(
        csv_path
        for csv_path in root.rglob("*.csv")
        if "log_file" not in csv_path.parts and "hw_file" not in csv_path.parts
    )


def _timestamp_seconds(line: str) -> int | None:
    match = TIME_RE.match(line)
    if match is None:
        return None
    return (
        int(match.group("hour")) * 3600
        + int(match.group("minute")) * 60
        + int(match.group("second"))
    )


def _parse_timed_lines(server_path: Path) -> list[tuple[int, str]]:
    with server_path.open("r", encoding="utf-8", errors="replace") as handle:
        return _parse_timed_lines_from_lines(handle)


def _parse_timed_lines_from_lines(lines: Iterable[str]) -> list[tuple[int, str]]:
    timed_lines: list[tuple[int, str]] = []
    current_timestamp: int | None = None
    rollover_offset = 0
    previous_raw_timestamp: int | None = None

    for line in lines:
        raw_timestamp = _timestamp_seconds(line)
        if raw_timestamp is not None:
            if previous_raw_timestamp is not None and raw_timestamp + rollover_offset < previous_raw_timestamp:
                rollover_offset += 24 * 3600
            current_timestamp = raw_timestamp + rollover_offset
            previous_raw_timestamp = current_timestamp
        if current_timestamp is None:
            continue
        timed_lines.append((current_timestamp, line))
    return timed_lines


def _select_lines(
    timed_lines: list[tuple[int, str]],
    timestamps: list[int],
    start: int,
    end: int,
) -> list[str]:
    left = bisect.bisect_left(timestamps, start)
    right = bisect.bisect_right(timestamps, end)
    return [line for _timestamp, line in timed_lines[left:right]]


def _align_window_to_timeline(start: int, end: int, timestamps: list[int]) -> tuple[int, int]:
    if not timestamps:
        return start, end

    day_seconds = 24 * 3600
    if end < start:
        end += day_seconds

    server_start = timestamps[0]
    server_end = timestamps[-1]
    server_midpoint = (server_start + server_end) / 2

    best_start = start
    best_end = end
    best_score: tuple[int, int, float] | None = None
    for day_index in range(server_start // day_seconds - 1, server_end // day_seconds + 2):
        candidate_start = start + day_index * day_seconds
        candidate_end = end + day_index * day_seconds
        overlap = max(0, min(candidate_end, server_end) - max(candidate_start, server_start) + 1)
        if overlap:
            gap = 0
        else:
            gap = min(
                abs(candidate_start - server_end),
                abs(candidate_end - server_start),
            )
        midpoint = (candidate_start + candidate_end) / 2
        score = (overlap, -gap, -abs(midpoint - server_midpoint))
        if best_score is None or score > best_score:
            best_score = score
            best_start = candidate_start
            best_end = candidate_end

    return best_start, best_end


def _looks_already_sliced(server_path: Path) -> bool:
    try:
        with server_path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline()
    except OSError:
        return False
    return first_line.startswith("# Sliced vLLM server log")


def _looks_combined_already_sliced(log_path: Path) -> bool:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            return any(line.startswith("# Sliced vLLM server log") for line in handle)
    except OSError:
        return False


def _slice_combined_log_file(
    log_path: Path,
    *,
    padding_seconds: int,
    backup: bool,
) -> str:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    client_section = _section_lines(lines, "CLIENT LOG")
    server_section = _section_lines(lines, "SERVER LOG")
    if not server_section:
        return "missing_server"

    try:
        start, end = _extract_client_window_from_lines(client_section or lines, log_path)
    except ValueError:
        return "window_parse_failed"

    server_body = _section_body_lines(server_section)
    timed_lines = _parse_timed_lines_from_lines(server_body)
    timestamps = [timestamp for timestamp, _line in timed_lines]
    aligned_start, aligned_end = _align_window_to_timeline(start, end, timestamps)
    sliced_lines = _select_lines(
        timed_lines,
        timestamps,
        aligned_start - padding_seconds,
        aligned_end + padding_seconds,
    )

    if backup:
        backup_path = log_path.with_suffix(log_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(log_path, backup_path)

    _write_combined_sliced_log(
        log_path,
        client_section or lines[: _section_start(lines, "SERVER LOG")],
        start,
        end,
        padding_seconds,
        sliced_lines,
    )
    return "sliced" if sliced_lines else "empty"


def _extract_client_window_from_lines(lines: Iterable[str], path: Path) -> tuple[int, int]:
    start: int | None = None
    end: int | None = None
    last_timestamp: int | None = None

    for line in lines:
        timestamp = _timestamp_seconds(line)
        if timestamp is not None:
            last_timestamp = timestamp
        if CLIENT_START_MARKER in line and timestamp is not None and start is None:
            start = timestamp
        if any(marker in line for marker in CLIENT_END_MARKERS) and timestamp is not None:
            end = timestamp

    if start is None:
        raise ValueError(f"Could not find client start marker in {path}")
    if end is None:
        end = last_timestamp
    if end is None:
        raise ValueError(f"Could not find client end timestamp in {path}")
    return start, end


def _section_start(lines: list[str], title: str) -> int:
    marker = f"===== {title} ====="
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return index
    return len(lines)


def _section_lines(lines: list[str], title: str) -> list[str]:
    start = _section_start(lines, title)
    if start == len(lines):
        return []
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("=====") and stripped.endswith("====="):
            end = index
            break
    return lines[start:end]


def _section_body_lines(section: list[str]) -> list[str]:
    for index, line in enumerate(section):
        if index == 0:
            continue
        if line.strip() == "":
            return section[index + 1 :]
    return section[1:]


def _write_sliced_log(
    server_path: Path,
    client_path: Path,
    start: int,
    end: int,
    padding_seconds: int,
    sliced_lines: list[str],
) -> None:
    header = [
        "# Sliced vLLM server log\n",
        f"# client_log={client_path.name}\n",
        f"# client_window={_format_seconds(start)}-{_format_seconds(end)}\n",
        f"# padding_seconds={padding_seconds}\n",
        f"# selected_lines={len(sliced_lines)}\n",
    ]
    if not sliced_lines:
        header.append("# no_server_lines_in_window=true\n")
    tmp_path = server_path.with_name(server_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(header)
        handle.writelines(sliced_lines)
    tmp_path.replace(server_path)


def _write_combined_sliced_log(
    log_path: Path,
    client_section: list[str],
    start: int,
    end: int,
    padding_seconds: int,
    sliced_lines: list[str],
) -> None:
    server_header = [
        "===== SERVER LOG =====\n",
        "# Sliced vLLM server log\n",
        f"# client_window={_format_seconds(start)}-{_format_seconds(end)}\n",
        f"# padding_seconds={padding_seconds}\n",
        f"# selected_lines={len(sliced_lines)}\n",
    ]
    if not sliced_lines:
        server_header.append("# no_server_lines_in_window=true\n")

    tmp_path = log_path.with_name(log_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(client_section)
        if client_section and not client_section[-1].endswith("\n"):
            handle.write("\n")
        handle.write("\n")
        handle.writelines(server_header)
        handle.writelines(sliced_lines)
    tmp_path.replace(log_path)


def _format_seconds(value: int) -> str:
    value %= 24 * 3600
    hour = value // 3600
    minute = (value % 3600) // 60
    second = value % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Slice normalized vLLM server logs to client-run windows.")
    parser.add_argument("--root", required=True, help="Normalized vLLM root directory.")
    parser.add_argument("--padding-seconds", type=int, default=3, help="Seconds to include before/after the client window.")
    parser.add_argument("--backup", action="store_true", help="Keep a .bak copy before overwriting each server log.")
    args = parser.parse_args()

    summary = slice_vllm_server_logs(
        args.root,
        padding_seconds=args.padding_seconds,
        backup=args.backup,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
