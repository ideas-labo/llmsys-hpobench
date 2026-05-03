import csv
import tempfile
import unittest
from pathlib import Path

from scripts.slice_vllm_server_logs import extract_client_window, slice_vllm_server_logs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, text: str) -> None:
    write_text(path, text)


class SliceVllmServerLogsTests(unittest.TestCase):
    def test_extract_client_window_uses_start_and_finish_markers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client_log = Path(temp_dir) / "client.log"
            write_text(
                client_log,
                """
[13:06:58] [Client-1-1] Starting vLLM Sampler Benchmark...
[13:07:00] [Client-1-1] progress
[13:08:07] [Client-1-1] === Client finished with exit code: 0 ===
""",
            )

            start, end = extract_client_window(client_log)

            self.assertEqual(start, 13 * 3600 + 6 * 60 + 58)
            self.assertEqual(end, 13 * 3600 + 8 * 60 + 7)

    def test_slices_server_log_to_client_window_with_padding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vLLM"
            fidelity = root / "5.0-0.5-4-50-r1"
            write_csv(
                fidelity / "5.0-0.5-4-50-r1.csv",
                """
ID,log-client-file,log-server-file
1,log_file/id1-client.log,log_file/id1-server.log
""",
            )
            write_text(
                fidelity / "log_file" / "id1-client.log",
                """
[13:06:58] [Client-1-1] Starting vLLM Sampler Benchmark...
[13:08:07] [Client-1-1] === Client finished with exit code: 0 ===
""",
            )
            write_text(
                fidelity / "log_file" / "id1-server.log",
                """
[13:06:40] [vLLM-1] init before window
[13:06:56] [vLLM-1] padding line
[13:06:58] [vLLM-1] first benchmark line
continuation belongs to first benchmark line
[13:08:07] [vLLM-1] final benchmark line
[13:08:11] [vLLM-1] after padding
""",
            )

            summary = slice_vllm_server_logs(root, padding_seconds=2)

            sliced = (fidelity / "log_file" / "id1-server.log").read_text(encoding="utf-8")
            self.assertIn("padding line", sliced)
            self.assertIn("first benchmark line", sliced)
            self.assertIn("continuation belongs", sliced)
            self.assertIn("final benchmark line", sliced)
            self.assertNotIn("init before window", sliced)
            self.assertNotIn("after padding", sliced)
            self.assertEqual(summary["server_logs_sliced"], 1)
            self.assertEqual(summary["missing_client_logs"], 0)

    def test_slicing_replaces_path_without_mutating_other_hardlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vLLM"
            fidelity = root / "5.0-0.5-4-50-r1"
            source_log = Path(temp_dir) / "shared-server.log"
            write_text(
                source_log,
                """
[13:06:40] [vLLM-1] init before window
[13:06:58] [vLLM-1] first benchmark line
[13:08:07] [vLLM-1] final benchmark line
[13:08:11] [vLLM-1] after window
""",
            )
            write_csv(
                fidelity / "5.0-0.5-4-50-r1.csv",
                """
ID,log-client-file,log-server-file
1,log_file/id1-client.log,log_file/id1-server.log
""",
            )
            write_text(
                fidelity / "log_file" / "id1-client.log",
                """
[13:06:58] [Client-1-1] Starting vLLM Sampler Benchmark...
[13:08:07] [Client-1-1] === Client finished with exit code: 0 ===
""",
            )
            (fidelity / "log_file").mkdir(parents=True, exist_ok=True)
            server_link = fidelity / "log_file" / "id1-server.log"
            server_link.hardlink_to(source_log)

            slice_vllm_server_logs(root, padding_seconds=0)

            self.assertIn("after window", source_log.read_text(encoding="utf-8"))
            self.assertNotIn("after window", server_link.read_text(encoding="utf-8"))

    def test_slicing_aligns_client_window_across_midnight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vLLM"
            fidelity = root / "5.0-0.5-4-50-r1"
            write_csv(
                fidelity / "5.0-0.5-4-50-r1.csv",
                """
ID,log-client-file,log-server-file
1,log_file/id1-client.log,log_file/id1-server.log
""",
            )
            write_text(
                fidelity / "log_file" / "id1-client.log",
                """
[00:45:56] [Client-1-1] Starting vLLM Sampler Benchmark...
[00:46:20] [Client-1-1] === Client finished with exit code: 0 ===
""",
            )
            write_text(
                fidelity / "log_file" / "id1-server.log",
                """
[23:04:59] [vLLM-1] previous day
[00:45:58] [vLLM-1] benchmark after midnight
[00:46:21] [vLLM-1] just outside padded window
""",
            )

            slice_vllm_server_logs(root, padding_seconds=0)

            sliced = (fidelity / "log_file" / "id1-server.log").read_text(encoding="utf-8")
            self.assertIn("benchmark after midnight", sliced)
            self.assertNotIn("previous day", sliced)
            self.assertNotIn("just outside", sliced)

    def test_slicing_empty_window_replaces_full_log_with_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vLLM"
            fidelity = root / "5.0-0.5-4-50-r1"
            write_csv(
                fidelity / "5.0-0.5-4-50-r1.csv",
                """
ID,log-client-file,log-server-file
1,log_file/id1-client.log,log_file/id1-server.log
""",
            )
            write_text(
                fidelity / "log_file" / "id1-client.log",
                """
[13:06:58] [Client-1-1] Starting vLLM Sampler Benchmark...
[13:08:07] [Client-1-1] === Client finished with exit code: 0 ===
""",
            )
            write_text(
                fidelity / "log_file" / "id1-server.log",
                """
[13:00:00] [vLLM-1] far before window
[13:20:00] [vLLM-1] far after window
""",
            )

            summary = slice_vllm_server_logs(root, padding_seconds=2)

            sliced = (fidelity / "log_file" / "id1-server.log").read_text(encoding="utf-8")
            self.assertIn("selected_lines=0", sliced)
            self.assertIn("no_server_lines_in_window=true", sliced)
            self.assertNotIn("far before window", sliced)
            self.assertNotIn("far after window", sliced)
            self.assertEqual(summary["empty_slices"], 1)


if __name__ == "__main__":
    unittest.main()
