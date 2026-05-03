import csv
import tempfile
import unittest
from pathlib import Path

from scripts.normalize_experiment_data import normalize_experiment_data


def write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class NormalizeExperimentDataTests(unittest.TestCase):
    def test_normalizes_headers_and_artifact_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-data"
            write_csv(
                root / "RAG" / "naiverag" / "f1" / "f1.csv",
                """
cfg-a,obj-MRR,obj-Test_Time,cost-total
x,0.7,12.5,3
y,0.8,11.5,4
""",
            )

            summary = normalize_experiment_data(root)

            header, rows = read_rows(root / "RAG" / "naiverag" / "f1" / "f1.csv")
            self.assertEqual(header[0], "ID")
            self.assertIn("obj-MRR+", header)
            self.assertIn("obj-Test_Time-", header)
            self.assertIn("hw-file", header)
            self.assertIn("log-client-file", header)
            self.assertIn("log-server-file", header)
            self.assertEqual(rows[0]["ID"], "1")
            self.assertEqual(rows[1]["ID"], "2")
            self.assertEqual(rows[0]["hw-file"], "")
            self.assertEqual(summary["csvs_rewritten"], 1)

    def test_removes_empty_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-data"
            write_csv(
                root / "RAG" / "LightRAG" / "f1" / "f1.csv",
                """
cfg-a,obj-precision,obj-test_time,
x,0.7,12.5,orphan
""",
            )

            normalize_experiment_data(root)

            header, rows = read_rows(root / "RAG" / "LightRAG" / "f1" / "f1.csv")
            self.assertNotIn("", header)
            self.assertIn("obj-precision+", header)
            self.assertIn("obj-test_time-", header)
            self.assertEqual(rows[0]["ID"], "1")

    def test_openhands_duplicate_csv_keeps_latest_and_renames_to_fidelity_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-data"
            fidelity_dir = root / "Agent" / "openhands" / "fc7_rc9_pd5_sc6"
            write_csv(
                fidelity_dir / "fc7_rc9_pd5_sc6.csv",
                """
cfg-a,obj-accuracy+
old,0
""",
            )
            write_csv(
                fidelity_dir / "5_1__fc7_rc9_pd5_sc6.csv",
                """
cfg-a,obj-accuracy+
new,1
""",
            )

            summary = normalize_experiment_data(root)

            target = fidelity_dir / "fc7_rc9_pd5_sc6.csv"
            self.assertTrue(target.is_file())
            self.assertFalse((fidelity_dir / "5_1__fc7_rc9_pd5_sc6.csv").exists())
            header, rows = read_rows(target)
            self.assertEqual(rows[0]["cfg-a"], "new")
            self.assertEqual(header[0], "ID")
            self.assertEqual(summary["openhands_duplicate_dirs_fixed"], 1)


if __name__ == "__main__":
    unittest.main()
