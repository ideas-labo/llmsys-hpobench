"""Normalize experiment-data CSVs to the project-wide benchmark format."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from typing import Iterable


ARTIFACT_DIRS = {"log_file", "hw_file"}
ARTIFACT_COLUMNS = ["hw-file", "log-file"]
LEGACY_LOG_COLUMNS = ("log-client-file", "log-server-file")
IGNORED_BENCHMARK_CSV_NAMES = {"sglang_multi_fidelity_benchmark_log.csv"}

OBJECTIVE_DIRECTIONS: dict[str, str] = {
    # RAG-style retrieval / answer quality metrics.
    "MRR": "+",
    "mrr": "+",
    "NDCG": "+",
    "ndcg": "+",
    "Context_Similarity": "+",
    "context_similarity": "+",
    "Lexical_AC": "+",
    "Answer_Precision": "+",
    "Answer_F1": "+",
    "LLM_AAJ": "+",
    "Answer_llmaaj": "+",
    "Avg_Similarity": "+",
    "precision": "+",
    "f1_score": "+",
    "recall": "+",
    "relevant_docs_count": "+",
    "is_successful": "+",
    # Lower is better.
    "Test_Time": "-",
    "test_time": "-",
    "best_match_position": "-",
}


def normalize_experiment_data(root: str | Path = "experiment-data") -> dict[str, int]:
    """Normalize all benchmark CSVs under ``root`` in place."""

    data_root = Path(root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"experiment data root not found: {data_root}")

    summary = {
        "csvs_rewritten": 0,
        "rows_rewritten": 0,
        "openhands_duplicate_dirs_fixed": 0,
        "empty_columns_removed": 0,
        "id_columns_added": 0,
        "artifact_columns_added": 0,
        "objective_columns_renamed": 0,
        "log_files_merged": 0,
        "fidelity_columns_removed": 0,
        "artifact_files_canonicalized": 0,
        "fidelity_dirs_renamed": 0,
    }

    summary["openhands_duplicate_dirs_fixed"] = _fix_openhands_duplicate_csv(data_root)
    summary["fidelity_dirs_renamed"] = _rename_legacy_fidelity_dirs(data_root)

    for csv_path in _benchmark_csvs(data_root):
        rewrite_summary = _normalize_csv(csv_path)
        for key, value in rewrite_summary.items():
            summary[key] += value

    return summary


def _fix_openhands_duplicate_csv(root: Path) -> int:
    fixed = 0
    openhands_root = root / "Agent" / "openhands"
    if not openhands_root.is_dir():
        return fixed

    for fidelity_dir in sorted(path for path in openhands_root.iterdir() if path.is_dir()):
        csvs = sorted(fidelity_dir.glob("*.csv"))
        if len(csvs) <= 1:
            continue

        preferred = fidelity_dir / f"5_1__{fidelity_dir.name}.csv"
        target = fidelity_dir / f"{fidelity_dir.name}.csv"
        if preferred.is_file():
            if target.exists():
                target.unlink()
            shutil.move(str(preferred), str(target))
            for extra_csv in sorted(fidelity_dir.glob("*.csv")):
                if extra_csv != target:
                    extra_csv.unlink()
            fixed += 1
    return fixed


def _rename_legacy_fidelity_dirs(root: Path) -> int:
    renamed = 0
    system_rules = {
        ("Agent", "openhands"): _canonical_openhands_fidelity_name,
        ("RAG", "LightRAG"): _replace_underscores_with_hyphens,
        ("RAG", "naiverag"): _replace_underscores_with_hyphens,
        ("RAG", "html_rag"): _canonical_html_rag_fidelity_name,
    }

    for (category, system), normalizer in system_rules.items():
        system_root = root / category / system
        if not system_root.is_dir():
            continue
        for fidelity_dir in sorted(path for path in system_root.iterdir() if path.is_dir()):
            new_name = normalizer(fidelity_dir.name)
            if not new_name or new_name == fidelity_dir.name:
                _rename_main_csv_to_match_dir(fidelity_dir, fidelity_dir.name)
                continue
            target_dir = fidelity_dir.with_name(new_name)
            if target_dir.exists():
                raise FileExistsError(f"cannot rename {fidelity_dir} to existing directory {target_dir}")
            _rename_main_csv_to_match_dir(fidelity_dir, new_name)
            shutil.move(str(fidelity_dir), str(target_dir))
            renamed += 1
    return renamed


def _rename_main_csv_to_match_dir(fidelity_dir: Path, fidelity_name: str) -> None:
    target = fidelity_dir / f"{fidelity_name}.csv"
    if target.exists():
        return
    csvs = sorted(fidelity_dir.glob("*.csv"))
    if len(csvs) == 1:
        shutil.move(str(csvs[0]), str(target))


def _canonical_openhands_fidelity_name(name: str) -> str:
    match = re.fullmatch(r"fc(\d+)_rc(\d+)_pd(\d+)_sc(\d+)", name)
    if not match:
        return name
    return "-".join(match.groups())


def _canonical_html_rag_fidelity_name(name: str) -> str:
    match = re.fullmatch(r"DC_([^_]+)_HR_([^_]+)_QR_([^_]+)", name)
    if not match:
        return name
    return "-".join(match.groups())


def _replace_underscores_with_hyphens(name: str) -> str:
    return name.replace("_", "-")


def _benchmark_csvs(root: Path) -> list[Path]:
    csvs: list[Path] = []
    for csv_path in root.rglob("*.csv"):
        if csv_path.name in IGNORED_BENCHMARK_CSV_NAMES:
            continue
        if any(part in ARTIFACT_DIRS for part in csv_path.parts):
            continue
        csvs.append(csv_path)
    return sorted(csvs)


def _normalize_csv(csv_path: Path) -> dict[str, int]:
    with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        original_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    normalized_fieldnames, field_mapping, stats = _normalize_header(original_fieldnames)
    normalized_rows = []
    for index, row in enumerate(rows, start=1):
        normalized_row = {field: "" for field in normalized_fieldnames}
        for original_field, value in row.items():
            target = field_mapping.get(original_field)
            if target is None:
                continue
            normalized_row[target] = value or ""
        if "ID" in normalized_fieldnames and not normalized_row.get("ID"):
            normalized_row["ID"] = str(index)
        if not normalized_row.get("log-file"):
            log_ref = _merge_legacy_log_file(
                csv_path.parent,
                normalized_row.get("ID") or str(index),
                row.get("log-client-file", ""),
                row.get("log-server-file", ""),
            )
            if log_ref:
                normalized_row["log-file"] = log_ref
                stats["log_files_merged"] += 1
        canonicalized = _canonicalize_artifact_files(
            csv_path.parent,
            normalized_row.get("ID") or str(index),
            normalized_row,
        )
        stats["artifact_files_canonicalized"] += canonicalized
        for artifact_column in ARTIFACT_COLUMNS:
            normalized_row.setdefault(artifact_column, "")
        normalized_rows.append(normalized_row)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=normalized_fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    stats["csvs_rewritten"] = 1
    stats["rows_rewritten"] = len(normalized_rows)
    return stats


def _normalize_header(fieldnames: Iterable[str]) -> tuple[list[str], dict[str, str | None], dict[str, int]]:
    stats = {
        "csvs_rewritten": 0,
        "rows_rewritten": 0,
        "empty_columns_removed": 0,
        "id_columns_added": 0,
        "artifact_columns_added": 0,
        "objective_columns_renamed": 0,
        "log_files_merged": 0,
        "fidelity_columns_removed": 0,
        "artifact_files_canonicalized": 0,
        "fidelity_dirs_renamed": 0,
    }

    normalized_fields: list[str] = []
    mapping: dict[str, str | None] = {}

    for original_field in fieldnames:
        if original_field == "":
            mapping[original_field] = None
            stats["empty_columns_removed"] += 1
            continue
        if original_field.startswith("FIDELITY_"):
            mapping[original_field] = None
            stats["fidelity_columns_removed"] += 1
            continue
        if original_field in LEGACY_LOG_COLUMNS:
            mapping[original_field] = None
            continue

        target = _normalize_column_name(original_field)
        if target != original_field and original_field.startswith("obj-"):
            stats["objective_columns_renamed"] += 1
        mapping[original_field] = target
        if target not in normalized_fields:
            normalized_fields.append(target)

    if not normalized_fields or normalized_fields[0] != "ID":
        normalized_fields = ["ID"] + [field for field in normalized_fields if field != "ID"]
        stats["id_columns_added"] += 1

    for artifact_column in ARTIFACT_COLUMNS:
        if artifact_column not in normalized_fields:
            normalized_fields.append(artifact_column)
            stats["artifact_columns_added"] += 1

    return normalized_fields, mapping, stats


def _normalize_column_name(column: str) -> str:
    if column == "id":
        return "ID"
    if not column.startswith("obj-"):
        return column
    if column.endswith("+") or column.endswith("-"):
        return column
    metric = column[len("obj-") :]
    direction = OBJECTIVE_DIRECTIONS.get(metric)
    if direction is None:
        direction = "-" if _looks_like_time_or_cost(metric) else "+"
    return f"{column}{direction}"


def _merge_legacy_log_file(fidelity_dir: Path, row_id: str, client_ref: str, server_ref: str) -> str:
    sections = []
    for title, ref in (("CLIENT LOG", client_ref), ("SERVER LOG", server_ref)):
        ref = (ref or "").strip()
        if not ref:
            continue
        source = fidelity_dir / ref
        if not source.is_file():
            continue
        sections.append((title, ref, source.read_text(encoding="utf-8", errors="replace")))

    if not sections:
        return ""

    log_dir = fidelity_dir / "log_file"
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"log-{row_id}.txt"
    with target.open("w", encoding="utf-8", newline="") as handle:
        for index, (title, ref, content) in enumerate(sections):
            if index:
                handle.write("\n")
            handle.write(f"===== {title} =====\n")
            handle.write(f"source: {ref}\n\n")
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
    return f"log_file/{target.name}"


def _canonicalize_artifact_files(fidelity_dir: Path, row_id: str, row: dict[str, str]) -> int:
    count = 0
    count += _canonicalize_artifact_file(
        fidelity_dir,
        row,
        column="log-file",
        artifact_dir="log_file",
        filename=f"log-{row_id}.txt",
    )
    count += _canonicalize_artifact_file(
        fidelity_dir,
        row,
        column="hw-file",
        artifact_dir="hw_file",
        filename=f"hw-{row_id}.txt",
    )
    return count


def _canonicalize_artifact_file(
    fidelity_dir: Path,
    row: dict[str, str],
    *,
    column: str,
    artifact_dir: str,
    filename: str,
) -> int:
    ref = (row.get(column) or "").strip()
    if not ref:
        return 0

    target = fidelity_dir / artifact_dir / filename
    target_ref = _relative_path(target, fidelity_dir)
    if ref == target_ref:
        return 0

    source = fidelity_dir / ref
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        row[column] = target_ref
        return 1

    if target.is_file():
        row[column] = target_ref
    return 0


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace(os.sep, "/")


def _looks_like_time_or_cost(metric: str) -> bool:
    lowered = metric.lower()
    return any(token in lowered for token in ("time", "latency", "duration", "deviation", "position"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize experiment-data CSVs in place.")
    parser.add_argument("--root", default="experiment-data", help="Experiment data root.")
    args = parser.parse_args()

    summary = normalize_experiment_data(args.root)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
