import csv
import tempfile
import unittest
from pathlib import Path

from scripts.normalize_vllm import normalize_vllm_dataset


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


class NormalizeVllmTests(unittest.TestCase):
    def test_normalizes_columns_splits_repeats_and_links_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "vLLM"
            output = root / "normalized" / "vLLM"
            write_text(
                source / "rate5.0_burst0.5_conc4_prompts50.csv",
                """
id,config_id,repeat,tp_size,max_num_seqs,enable_prefix_caching,enable_speculative_decoding,speculative_method,num_speculative_tokens,prompt_lookup_max,temperature,top_k,min_p,repetition_penalty,length_penalty,benchmark_duration_s,request_throughput,mean_ttft_ms,gpu_kv_cache_usage_avg,process_cpu_seconds_avg
1,10,1,1,1024,True,True,ngram,3,6,0.2,50,0.1,1.1,1.0,10.5,4.2,12.3,0.55,1.5
2,11,1,1,2048,False,False,disabled,0,0,0.7,80,0.3,1.2,1.1,11.5,3.2,22.3,0.65,1.7
3,10,2,1,1024,True,True,ngram,3,6,0.2,50,0.1,1.1,1.0,12.5,5.2,11.3,0.75,1.9
""",
            )
            write_text(source / "logs" / "client_config_10_fidelity_1_20250101.log", "client one")
            write_text(source / "logs" / "client_config_10_fidelity_2_20250101.log", "client two")
            write_text(source / "logs" / "server_config_10_20250101.log", "server ten")

            summary = normalize_vllm_dataset(source, output, artifact_mode="copy")

            self.assertEqual(summary["input_csv_files"], 1)
            self.assertEqual(summary["output_csv_files"], 2)
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["linked_client_logs"], 2)
            self.assertEqual(summary["linked_server_logs"], 2)
            self.assertEqual(summary["linked_hardware_files"], 0)

            repeat_one_csv = output / "5.0-0.5-4-50-r1" / "5.0-0.5-4-50-r1.csv"
            repeat_two_csv = output / "5.0-0.5-4-50-r2" / "5.0-0.5-4-50-r2.csv"
            self.assertTrue(repeat_one_csv.is_file())
            self.assertTrue(repeat_two_csv.is_file())

            with repeat_one_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ID"], "1")
            self.assertEqual(rows[0]["cfg-config_id"], "10")
            self.assertNotIn("FIDELITY_rate", rows[0])
            self.assertNotIn("FIDELITY_repeat", rows[0])
            self.assertEqual(rows[0]["cfg-max_num_seqs"], "1024")
            self.assertEqual(rows[0]["cfg-ai-enable_prefix_caching"], "True")
            self.assertEqual(rows[0]["cfg-ai-temperature"], "0.2")
            self.assertEqual(rows[0]["obj-request_throughput+"], "4.2")
            self.assertEqual(rows[0]["obj-mean_ttft_ms-"], "12.3")
            self.assertEqual(rows[0]["cost-benchmark_duration_s"], "10.5")
            self.assertEqual(rows[0]["cost-gpu_kv_cache_usage_avg"], "0.55")
            self.assertEqual(rows[0]["log-file"], "log_file/id1.log")
            self.assertEqual(rows[0]["hw-file"], "")
            self.assertEqual(rows[1]["log-file"], "")
            self.assertEqual(rows[1]["hw-file"], "")
            self.assertTrue((repeat_one_csv.parent / rows[0]["log-file"]).is_file())
            merged_log = (repeat_one_csv.parent / rows[0]["log-file"]).read_text(encoding="utf-8")
            self.assertIn("===== CLIENT LOG =====", merged_log)
            self.assertIn("client one", merged_log)
            self.assertIn("===== SERVER LOG =====", merged_log)
            self.assertIn("server ten", merged_log)

            with repeat_two_csv.open("r", newline="", encoding="utf-8") as handle:
                repeat_two_rows = list(csv.DictReader(handle))

            self.assertEqual(repeat_two_rows[0]["ID"], "3")
            self.assertEqual(repeat_two_rows[0]["log-file"], "log_file/id3.log")
            self.assertIn("client two", (repeat_two_csv.parent / repeat_two_rows[0]["log-file"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
