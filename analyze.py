import argparse
import csv
from pathlib import Path


def _detect_header(sample_text: str) -> bool:
	try:
		return csv.Sniffer().has_header(sample_text)
	except csv.Error:
		return False


def count_csv_rows(csv_path: Path, header_mode: str) -> int:
	if not csv_path.is_file():
		return 0

	with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
		sample = handle.read(2048)
		handle.seek(0)

		if not sample.strip():
			return 0

		reader = csv.reader(handle)
		skip_header = False
		if header_mode == "yes":
			skip_header = True
		elif header_mode == "auto":
			skip_header = _detect_header(sample)

		if skip_header:
			next(reader, None)

		return sum(1 for _ in reader)


def summarize_system(system_dir: Path, header_mode: str) -> dict:
	workload_counts = {}
	for workload_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
		csv_files = sorted(workload_dir.rglob("*.csv"))
		if not csv_files:
			continue

		total_rows = sum(count_csv_rows(csv_path, header_mode) for csv_path in csv_files)
		workload_counts[workload_dir.name] = total_rows

	root_csv_files = sorted(system_dir.glob("*.csv"))
	for csv_path in root_csv_files:
		workload_name = csv_path.stem
		workload_counts[workload_name] = count_csv_rows(csv_path, header_mode)

	total_samples = sum(workload_counts.values())

	if workload_counts:
		counts = list(workload_counts.values())
		min_count = min(counts)
		max_count = max(counts)
	else:
		min_count = 0
		max_count = 0

	return {
		"system": system_dir.name,
		"workloads": workload_counts,
		"min_count": min_count,
		"max_count": max_count,
		"total_samples": total_samples,
	}


def find_system_dirs(root_dir: Path) -> list[Path]:
	candidates = [p for p in root_dir.iterdir() if p.is_dir()]
	system_dirs = []
	for candidate in candidates:
		has_csv = any(candidate.rglob("*.csv"))
		if has_csv:
			system_dirs.append(candidate)
	return sorted(system_dirs)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Summarize sampled configurations and total sample points per system."
		)
	)
	parser.add_argument(
		"-r",
		"--root",
		default=str(Path.cwd()),
		help="Root directory that contains system folders.",
	)
	parser.add_argument(
		"--header",
		choices=["auto", "yes", "no"],
		default="auto",
		help="CSV header handling: auto|yes|no.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print per-workload counts.",
	)

	args = parser.parse_args()
	root_dir = Path(args.root).resolve()

	system_dirs = find_system_dirs(root_dir)
	if not system_dirs:
		print(f"No system folders with CSV files found under: {root_dir}")
		return 1

	for system_dir in system_dirs:
		summary = summarize_system(system_dir, args.header)
		workload_total = len(summary["workloads"])
		if workload_total == 0:
			print(f"{summary['system']}: no workloads with CSV files")
			continue

		if summary["min_count"] == summary["max_count"]:
			range_text = str(summary["min_count"])
		else:
			range_text = f"{summary['min_count']} - {summary['max_count']}"

		print(
			f"{summary['system']}: workload config range = {range_text} "
			f"(workloads: {workload_total}), total samples = {summary['total_samples']}"
		)

		if args.verbose:
			for workload_name, count in summary["workloads"].items():
				print(f"  - {workload_name}: {count}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
