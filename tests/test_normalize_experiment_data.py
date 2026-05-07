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
FIDELITY_factor,cfg-a,obj-MRR,obj-Test_Time,cost-total
f1,x,0.7,12.5,3
f1,y,0.8,11.5,4
""",
            )

            summary = normalize_experiment_data(root)

            header, rows = read_rows(root / "RAG" / "naiverag" / "f1" / "f1.csv")
            self.assertEqual(header[0], "ID")
            self.assertIn("obj-MRR+", header)
            self.assertIn("obj-Test_Time-", header)
            self.assertIn("hw-file", header)
            self.assertIn("log-file", header)
            self.assertNotIn("log-client-file", header)
            self.assertNotIn("log-server-file", header)
            self.assertNotIn("FIDELITY_factor", header)
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

            renamed_dir = root / "Agent" / "openhands" / "7-9-5-6"
            target = renamed_dir / "7-9-5-6.csv"
            self.assertTrue(target.is_file())
            self.assertFalse((fidelity_dir / "5_1__fc7_rc9_pd5_sc6.csv").exists())
            header, rows = read_rows(target)
            self.assertEqual(rows[0]["cfg-a"], "new")
            self.assertEqual(header[0], "ID")
            self.assertEqual(summary["openhands_duplicate_dirs_fixed"], 1)
            self.assertEqual(summary["fidelity_dirs_renamed"], 1)

    def test_renames_legacy_fidelity_directories_to_hyphen_separated_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-data"
            cases = [
                ("RAG", "LightRAG", "bridge_0_0_2", "bridge-0-0-2"),
                ("RAG", "naiverag", "0.2_0_easy_agriculture", "0.2-0-easy-agriculture"),
                ("RAG", "html_rag", "DC_1_HR_01_QR_05", "1-01-05"),
                ("Agent", "openhands", "fc7_rc7_pd1_sc1", "7-7-1-1"),
            ]
            for category, system, old_name, _new_name in cases:
                write_csv(
                    root / category / system / old_name / f"{old_name}.csv",
                    """
cfg-a,obj-score+
x,1
""",
                )

            summary = normalize_experiment_data(root)

            self.assertEqual(summary["fidelity_dirs_renamed"], len(cases))
            for category, system, old_name, new_name in cases:
                self.assertFalse((root / category / system / old_name).exists())
                self.assertTrue((root / category / system / new_name / f"{new_name}.csv").is_file())

    def test_merges_legacy_client_and_server_logs_into_canonical_log_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-data"
            fidelity_dir = root / "Engine" / "vLLM" / "f1"
            write_csv(
                fidelity_dir / "f1.csv",
                """
ID,cfg-a,obj-score+,hw-file,log-client-file,log-server-file
1,1,0.5,,log_file/id1-client.log,log_file/id1-server.log
2,2,0.6,,log_file/id2-client.log,
""",
            )
            (fidelity_dir / "log_file").mkdir(parents=True, exist_ok=True)
            (fidelity_dir / "log_file" / "id1-client.log").write_text("client one\n", encoding="utf-8")
            (fidelity_dir / "log_file" / "id1-server.log").write_text("server one\n", encoding="utf-8")
            (fidelity_dir / "log_file" / "id2-client.log").write_text("client two\n", encoding="utf-8")

            summary = normalize_experiment_data(root)

            header, rows = read_rows(fidelity_dir / "f1.csv")
            self.assertIn("log-file", header)
            self.assertNotIn("log-client-file", header)
            self.assertNotIn("log-server-file", header)
            self.assertEqual(rows[0]["log-file"], "log_file/log-1.txt")
            self.assertEqual(rows[1]["log-file"], "log_file/log-2.txt")
            merged = (fidelity_dir / rows[0]["log-file"]).read_text(encoding="utf-8")
            self.assertIn("===== CLIENT LOG =====", merged)
            self.assertIn("client one", merged)
            self.assertIn("===== SERVER LOG =====", merged)
            self.assertIn("server one", merged)
            client_only = (fidelity_dir / rows[1]["log-file"]).read_text(encoding="utf-8")
            self.assertIn("===== CLIENT LOG =====", client_only)
            self.assertNotIn("===== SERVER LOG =====", client_only)
            self.assertEqual(summary["log_files_merged"], 2)

    def test_canonicalizes_existing_artifact_file_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-data"
            fidelity_dir = root / "Engine" / "SGLang" / "f1"
            write_csv(
                fidelity_dir / "f1.csv",
                """
ID,cfg-a,obj-score+,hw-file,log-file
1,1,0.5,hw_file/id1-hw.csv,log_file/failed_exit-3_20250101.log
""",
            )
            (fidelity_dir / "hw_file").mkdir(parents=True, exist_ok=True)
            (fidelity_dir / "log_file").mkdir(parents=True, exist_ok=True)
            (fidelity_dir / "hw_file" / "id1-hw.csv").write_text("gpu\n", encoding="utf-8")
            (fidelity_dir / "log_file" / "failed_exit-3_20250101.log").write_text("failed\n", encoding="utf-8")

            summary = normalize_experiment_data(root)

            _, rows = read_rows(fidelity_dir / "f1.csv")
            self.assertEqual(rows[0]["hw-file"], "hw_file/hw-1.txt")
            self.assertEqual(rows[0]["log-file"], "log_file/log-1.txt")
            self.assertTrue((fidelity_dir / "hw_file" / "hw-1.txt").is_file())
            self.assertTrue((fidelity_dir / "log_file" / "log-1.txt").is_file())
            self.assertFalse((fidelity_dir / "log_file" / "failed_exit-3_20250101.log").exists())
            self.assertEqual(summary["artifact_files_canonicalized"], 2)


if __name__ == "__main__":
    unittest.main()
