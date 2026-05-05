import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.normalize_sglang import include_sglang_failed_log_samples, normalize_sglang_dataset


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class NormalizeSglangTests(unittest.TestCase):
    def test_normalizes_json_samples_to_tab_format_order_and_combined_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "SGLang"
            raw_dir = root / "rate5.0_conc16_groups16_ppg8_syslen1024_qlen128_olen256"
            write_json(
                raw_dir / "tp1_pp1_reqs640_tokens20480_chunk10240_config1_fidelity1_20260321_210848.json",
                {
                    "backend": "sglang",
                    "dataset_name": "generated-shared-prefix",
                    "request_rate": 5.0,
                    "max_concurrency": 16,
                    "duration": 12.5,
                    "completed": 128,
                    "total_input_tokens": 1024,
                    "total_output_tokens": 2048,
                    "total_output_tokens_retokenized": 2048,
                    "request_throughput": 10.1,
                    "input_throughput": 20.2,
                    "output_throughput": 30.3,
                    "mean_e2e_latency_ms": 40.4,
                    "median_e2e_latency_ms": 41.4,
                    "std_e2e_latency_ms": 2.5,
                    "p99_e2e_latency_ms": 60.6,
                    "mean_ttft_ms": 7.7,
                    "median_ttft_ms": 8.8,
                    "std_ttft_ms": 1.2,
                    "p99_ttft_ms": 9.9,
                    "mean_tpot_ms": 3.3,
                    "median_tpot_ms": 3.4,
                    "std_tpot_ms": 0.4,
                    "p99_tpot_ms": 4.4,
                    "mean_itl_ms": 5.5,
                    "median_itl_ms": 5.6,
                    "std_itl_ms": 0.6,
                    "p95_itl_ms": 6.5,
                    "p99_itl_ms": 7.5,
                    "concurrency": 15.8,
                    "generated_texts": ["large raw field excluded from csv"],
                },
            )
            write_text(
                root / "logs" / "client_config_1_fidelity_1_20260321_210848.log",
                """
SGLang Benchmark Client Log (Shared-Prefix Dataset)
Configuration 1, Fidelity 1
Started at: 2026-03-21 21:08:48

SGLang Config:
{
  "tp_size": 1,
  "pp_size": 1,
  "max_running_requests": 640,
  "max_total_tokens": 20480,
  "chunked_prefill_size": 10240,
  "gpu_memory_utilization": 0.85,
  "attention_backend": "flashinfer",
  "context_length": 8192,
  "enable_torch_compile": true,
  "enable_p2p_check": true,
  "disable_radix_cache": false
}

Sampling Params:
{
  "temperature": 0.9,
  "top_k": 51,
  "top_p": 0.55,
  "repetition_penalty": 1.2,
  "frequency_penalty": 2.0
}

Completed at: 2026-03-21 21:09:05
""",
            )
            write_text(
                root / "logs" / "server_config_1_20260321_210800.log",
                """
STDERR: [2026-03-21 21:08:47] before sample
STDERR: [2026-03-21 21:08:49] inside sample
STDERR: [2026-03-21 21:09:06] after sample
STDERR: [2026-03-21 21:10:00] outside sample
""",
            )

            summary = normalize_sglang_dataset(root)

            self.assertEqual(summary["input_json_files"], 1)
            self.assertEqual(summary["output_csv_files"], 1)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["linked_client_logs"], 1)
            self.assertEqual(summary["linked_server_logs"], 1)

            csv_path = root / "5.0-1.0-16-16-1024" / "5.0-1.0-16-16-1024.csv"
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ID"], "1")
            self.assertEqual(rows[0]["cfg-config_id"], "1")
            self.assertEqual(rows[0]["cfg-max_running_requests"], "640")
            self.assertEqual(rows[0]["cfg-ai-temperature"], "0.9")
            self.assertEqual(rows[0]["obj-request_throughput+"], "10.1")
            self.assertEqual(rows[0]["obj-mean_ttft_ms-"], "7.7")
            self.assertEqual(rows[0]["cost-duration"], "12.5")
            self.assertEqual(rows[0]["hw-file"], "")
            self.assertEqual(rows[0]["log-file"], "log_file/id1.log")
            self.assertNotIn("FIDELITY_request_rate", rows[0])
            self.assertNotIn("generated_texts", rows[0])

            merged_log = (csv_path.parent / rows[0]["log-file"]).read_text(encoding="utf-8")
            self.assertIn("===== CLIENT LOG =====", merged_log)
            self.assertIn("SGLang Benchmark Client Log", merged_log)
            self.assertIn("===== SERVER LOG =====", merged_log)
            self.assertIn("selected_lines: 3", merged_log)
            self.assertIn("inside sample", merged_log)
            self.assertNotIn("outside sample", merged_log)

    def test_includes_exit_code_minus_three_log_rows_as_failed_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "SGLang"
            fidelity_dir = root / "5.0-1.0-16-16-1024"
            write_text(
                fidelity_dir / "5.0-1.0-16-16-1024.csv",
                """
ID,cfg-config_id,cfg-tp_size,cfg-pp_size,cfg-max_running_requests,cfg-max_total_tokens,cfg-chunked_prefill_size,cfg-gpu_memory_utilization,cfg-attention_backend,cfg-context_length,cfg-enable_torch_compile,cfg-enable_p2p_check,cfg-disable_radix_cache,cfg-ai-temperature,cfg-ai-top_k,cfg-ai-top_p,cfg-ai-repetition_penalty,cfg-ai-frequency_penalty,cost-duration,cost-total_input_tokens,cost-total_output_tokens,cost-total_output_tokens_retokenized,cost-concurrency,obj-completed+,obj-request_throughput+,obj-input_throughput+,obj-output_throughput+,obj-mean_e2e_latency_ms-,obj-median_e2e_latency_ms-,obj-std_e2e_latency_ms-,obj-p99_e2e_latency_ms-,obj-mean_ttft_ms-,obj-median_ttft_ms-,obj-std_ttft_ms-,obj-p99_ttft_ms-,obj-mean_tpot_ms-,obj-median_tpot_ms-,obj-std_tpot_ms-,obj-p99_tpot_ms-,obj-mean_itl_ms-,obj-median_itl_ms-,obj-std_itl_ms-,obj-p95_itl_ms-,obj-p99_itl_ms-,hw-file,log-file
1,1,1,1,640,20480,10240,0.85,flashinfer,8192,True,True,False,0.9,51,0.55,1.2,2.0,31.5,154032,32768,32765,15.6,128,4.0,4878,1037,,,,,,,,,,,,,,,,,,log_file/id1.log
""",
            )
            write_text(
                root / "sglang_multi_fidelity_benchmark_log.csv",
                """
tp_size,pp_size,max_running_requests,max_total_tokens,chunked_prefill_size,gpu_memory_utilization,attention_backend,context_length,enable_torch_compile,enable_p2p_check,disable_radix_cache,extra_body,request_rate,max_concurrency,burstiness,gsp_num_groups,gsp_prompts_per_group,gsp_system_prompt_len,gsp_question_len,gsp_output_len,exit_code,duration,completed,total_input_tokens,total_output_tokens,request_throughput,output_throughput,mean_ttft_ms,median_ttft_ms,mean_itl_ms,median_itl_ms,p99_itl_ms,timestamp
1,1,832,24576,11264,0.75,torch_native,13312,True,False,False,"{""temperature"": 0.5, ""top_k"": 91, ""top_p"": 0.65, ""repetition_penalty"": 1.8, ""frequency_penalty"": 0.9}",5.0,16,1.0,16,8,1024,128,256,-3,,0,,,,,,,,,,20260502_024912
""",
            )

            summary = include_sglang_failed_log_samples(root)

            self.assertEqual(summary["failed_log_rows"], 1)
            self.assertEqual(summary["appended_failed_rows"], 1)
            csv_path = fidelity_dir / "5.0-1.0-16-16-1024.csv"
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            failed = rows[1]
            self.assertEqual(failed["ID"], "2")
            self.assertEqual(failed["cfg-config_id"], "failed-2")
            self.assertEqual(failed["cfg-context_length"], "13312")
            self.assertEqual(failed["cfg-ai-temperature"], "0.5")
            self.assertEqual(failed["cost-duration"], "")
            self.assertEqual(failed["obj-completed+"], "0")
            self.assertEqual(failed["log-file"], "log_file/failed_exit-3_20260502_024912_2.log")

            failed_log = (fidelity_dir / failed["log-file"]).read_text(encoding="utf-8")
            self.assertIn("Skip config 2: context_length=13312 exceeds 8192", failed_log)
            self.assertIn("Result saved: exit_code=-3", failed_log)

            second_summary = include_sglang_failed_log_samples(root)
            self.assertEqual(second_summary["appended_failed_rows"], 0)


if __name__ == "__main__":
    unittest.main()
