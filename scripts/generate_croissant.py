"""Generate Croissant metadata for LLMSYS-HPOBench."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_DIRS = {"log_file", "hw_file"}
IGNORED_CSV_NAMES = {"sglang_multi_fidelity_benchmark_log.csv"}
MANIFEST_COLUMNS = [
    "row_key",
    "system",
    "category",
    "fidelity",
    "record_id",
    "config_id",
    "csv_file",
    "hw_file",
    "log_file",
]


def generate_croissant(
    *,
    data_root: str | Path = "experiment-data",
    output_path: str | Path = "croissant.json",
    records_output_path: str | Path = "metadata/croissant_records.csv",
    dataset_name: str = "LLMSYS-HPOBench",
    dataset_url: str = "https://github.com/TODO/LLMSYS-HPOBench",
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
    creators: Iterable[str] = ("LLMSYS-HPOBench Authors",),
) -> dict[str, int]:
    """Generate a Croissant JSON-LD file and a sample manifest CSV."""

    data_root_path = Path(data_root).resolve()
    output = Path(output_path).resolve()
    records_output = Path(records_output_path).resolve()
    if not data_root_path.is_dir():
        raise FileNotFoundError(f"dataset root not found: {data_root_path}")

    records, missing_artifacts = _collect_records(data_root_path)
    records_output.parent.mkdir(parents=True, exist_ok=True)
    with records_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    metadata = _build_croissant_metadata(
        dataset_name=dataset_name,
        dataset_url=dataset_url,
        license_url=license_url,
        creators=list(creators),
        data_root=data_root_path,
        records_output=records_output,
        output=output,
        record_count=len(records),
        systems=sorted({record["system"] for record in records}),
        fidelity_count=len({(record["system"], record["fidelity"]) for record in records}),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "rows": len(records),
        "systems": len({record["system"] for record in records}),
        "fidelities": len({(record["system"], record["fidelity"]) for record in records}),
        "missing_artifacts": missing_artifacts,
        "croissant_json_bytes": output.stat().st_size,
        "manifest_csv_bytes": records_output.stat().st_size,
    }


def _collect_records(data_root: Path) -> tuple[list[dict[str, str]], int]:
    records: list[dict[str, str]] = []
    missing_artifacts = 0

    for csv_path in _benchmark_csvs(data_root):
        relative_parts = csv_path.relative_to(data_root).parts
        if len(relative_parts) < 4:
            continue
        category, system = relative_parts[0], relative_parts[1]
        fidelity = csv_path.stem
        with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record_id = (row.get("ID") or "").strip()
                hw_file = _artifact_path(data_root, csv_path, row.get("hw-file", ""))
                log_file = _artifact_path(data_root, csv_path, row.get("log-file", ""))
                if (row.get("hw-file") and not hw_file) or (row.get("log-file") and not log_file):
                    missing_artifacts += 1
                records.append(
                    {
                        "row_key": f"{system}/{fidelity}/{record_id}",
                        "system": system,
                        "category": category,
                        "fidelity": fidelity,
                        "record_id": record_id,
                        "config_id": _config_id(row),
                        "csv_file": _relative_path(csv_path, data_root),
                        "hw_file": hw_file,
                        "log_file": log_file,
                    }
                )

    return records, missing_artifacts


def _benchmark_csvs(data_root: Path) -> list[Path]:
    csvs = []
    for csv_path in data_root.rglob("*.csv"):
        if csv_path.name in IGNORED_CSV_NAMES:
            continue
        if any(part in ARTIFACT_DIRS for part in csv_path.parts):
            continue
        if csv_path.stem != csv_path.parent.name:
            continue
        csvs.append(csv_path)
    return sorted(csvs)


def _artifact_path(data_root: Path, csv_path: Path, ref: str) -> str:
    ref = (ref or "").strip()
    if not ref:
        return ""
    path = csv_path.parent / ref
    if not path.is_file():
        return ""
    return _relative_path(path, data_root)


def _config_id(row: dict[str, str]) -> str:
    for key in ("cfg-config_id", "cfg-agent_config_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _build_croissant_metadata(
    *,
    dataset_name: str,
    dataset_url: str,
    license_url: str,
    creators: list[str],
    data_root: Path,
    records_output: Path,
    output: Path,
    record_count: int,
    systems: list[str],
    fidelity_count: int,
) -> dict[str, Any]:
    records_ref = _relative_path(records_output, output.parent)
    data_root_ref = _relative_path(data_root, output.parent)
    manifest_id = "croissant_records_csv"

    return {
        "@context": _croissant_context(),
        "@type": "sc:Dataset",
        "name": dataset_name,
        "description": (
            "LLMSYS-HPOBench is an offline benchmark dataset for hyperparameter "
            "optimization of LLM systems, covering inference engines, RAG pipelines, "
            "and agent frameworks. Each row links normalized objective/cost values "
            "to the corresponding log and hardware artifacts when available."
        ),
        "url": dataset_url,
        "license": license_url,
        "version": "1.0.0",
        "datePublished": datetime.now(timezone.utc).date().isoformat(),
        "citeAs": "TODO: add the NeurIPS 2026 dataset paper citation or DOI before submission.",
        "creator": [{"@type": "Person", "name": creator} for creator in creators],
        "keywords": [
            "LLM systems",
            "hyperparameter optimization",
            "benchmark",
            "inference engine",
            "RAG",
            "agents",
        ],
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "dct:conformsTo": ["http://mlcommons.org/croissant/1.0", "http://mlcommons.org/croissant/RAI/1.0"],
        "rai:dataUseCases": [
            "Offline benchmarking of LLM-system hyperparameter optimization methods.",
            "Analysis of trade-offs among system configuration, AI parameters, performance, cost, logs, and hardware traces.",
        ],
        "rai:dataLimitations": (
            "The dataset contains observed benchmark samples rather than a complete "
            "continuous configuration space. Missing config-fidelity combinations "
            "should be treated as unobserved, not as failed runs unless explicitly "
            "encoded in the normalized CSV."
        ),
        "rai:personalSensitiveInformation": (
            "The normalized benchmark data is intended to contain system metrics, "
            "synthetic or benchmark task outputs, logs, and hardware measurements. "
            "Contributors should review logs before publication to remove any "
            "accidental secrets, credentials, or personal data."
        ),
        "rai:dataCollection": (
            "Samples were collected from local benchmark workflows and normalized "
            "into a common CSV schema with per-row log and hardware artifact references."
        ),
        "rai:dataPreprocessing": (
            "Raw outputs were cleaned into prefixed columns: cfg-* for non-AI parameters, "
            "cfg-ai-* for AI parameters, obj-* for objective metrics, cost-* for cost metrics, "
            "hw-file for hardware artifacts, and log-file for logs."
        ),
        "rai:maintenancePlan": (
            "Future systems can be added by contributing normalized fidelity directories, "
            "system manuals, and tests following CONTRIBUTING.md."
        ),
        "rai:dataSocialImpact": (
            "The dataset is designed to support reproducible evaluation of LLM-system "
            "optimization methods and to make cost/performance trade-offs more transparent."
        ),
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": manifest_id,
                "name": records_output.name,
                "description": "Manifest with one row per normalized benchmark sample.",
                "contentUrl": records_ref,
                "encodingFormat": "text/csv",
                "sha256": _sha256(records_output),
            },
            {
                "@type": "cr:FileSet",
                "@id": "normalized_benchmark_csv_files",
                "name": "Normalized benchmark CSV files",
                "description": "One main CSV per system fidelity directory.",
                "contentUrl": data_root_ref,
                "encodingFormat": "text/csv",
                "includes": "**/*.csv",
                "excludes": ["**/log_file/**", "**/hw_file/**", "**/raw_metadata/**"],
            },
            {
                "@type": "cr:FileSet",
                "@id": "log_artifacts",
                "name": "Per-sample log artifacts",
                "description": "Per-row logs referenced by the log-file column.",
                "contentUrl": data_root_ref,
                "encodingFormat": "text/plain",
                "includes": "**/log_file/log-*.txt",
            },
            {
                "@type": "cr:FileSet",
                "@id": "hardware_artifacts",
                "name": "Per-sample hardware artifacts",
                "description": "Per-row hardware traces referenced by the hw-file column.",
                "contentUrl": data_root_ref,
                "encodingFormat": "text/plain",
                "includes": "**/hw_file/hw-*.txt",
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "sample_manifest",
                "name": "sample_manifest",
                "description": (
                    f"Manifest of {record_count} normalized benchmark samples across "
                    f"{len(systems)} systems and {fidelity_count} system-fidelity directories."
                ),
                "key": [{"@id": "sample_manifest/row_key"}],
                "field": _manifest_fields(manifest_id),
            }
        ],
    }


def _manifest_fields(file_object_id: str) -> list[dict[str, Any]]:
    fields = []
    for column in MANIFEST_COLUMNS:
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"sample_manifest/{column}",
                "name": column,
                "dataType": "sc:Text",
                "source": {
                    "fileObject": file_object_id,
                    "extract": {"column": column},
                },
            }
        )
    return fields


def _croissant_context() -> dict[str, Any]:
    return {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "equivalentProperty": "cr:equivalentProperty",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "samplingRate": "cr:samplingRate",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace(os.sep, "/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Croissant metadata for LLMSYS-HPOBench.")
    parser.add_argument("--root", default="experiment-data", help="Normalized dataset root.")
    parser.add_argument("--output", default="croissant.json", help="Output Croissant JSON-LD path.")
    parser.add_argument(
        "--records-output",
        default="metadata/croissant_records.csv",
        help="Output manifest CSV path.",
    )
    parser.add_argument("--dataset-url", default="https://github.com/TODO/LLMSYS-HPOBench")
    parser.add_argument("--license-url", default="https://creativecommons.org/licenses/by/4.0/")
    parser.add_argument(
        "--creator",
        action="append",
        dest="creators",
        default=None,
        help="Creator name. May be repeated.",
    )
    args = parser.parse_args()

    summary = generate_croissant(
        data_root=args.root,
        output_path=args.output,
        records_output_path=args.records_output,
        dataset_url=args.dataset_url,
        license_url=args.license_url,
        creators=args.creators or ["LLMSYS-HPOBench Authors"],
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
