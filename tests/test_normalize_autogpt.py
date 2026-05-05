import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.normalize_autogpt import normalize_autogpt_dataset


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class NormalizeAutoGptTests(unittest.TestCase):
    def test_normalizes_samples_to_tab_order_and_links_logs_and_hardware(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "autogpt"
            raw_fidelity = root / "large_scale" / "fidelities" / "simple_r1_code_generation"
            sample_name = "agent_config_1"
            write_json(
                raw_fidelity / f"{sample_name}.json",
                {
                    "sample_id": "agent_config_1_simple_r1_code_generation",
                    "agent_config_id": "agent_config_1",
                    "fidelity_config": {
                        "model_name": "qwen2.5-7b-instruct",
                        "big_brain": True,
                        "use_functions_api": False,
                        "cycle_budget": 3,
                        "send_token_limit": 2048,
                        "cognitive_strategy": "one_shot",
                        "enabled_components": ["FileManagerComponent", "CodeExecutorComponent"],
                        "temperature": 0.7,
                        "max_tokens": 512,
                        "prompt_style": "concise",
                        "allow_fs_access": True,
                        "full_message_count": 4,
                        "shell_command_control": "allowlist",
                        "task_type": "simple",
                        "requests_count": 1,
                        "workload_category": "code_generation",
                        "information_availability": "mixed",
                    },
                    "agent_config": {
                        "execution_constraints": {
                            "task_max_duration_s": 180.0,
                        }
                    },
                    "num_runs": 1,
                    "metrics": {
                        "success_rate": 1.0,
                        "correctness_score": 0.5,
                        "instruction_adherence_score": 1.0,
                        "error_rate": 0.0,
                        "timeout_rate": 0.0,
                        "total_duration": 12.5,
                        "avg_task_duration": 12.4,
                        "total_tokens": 100,
                        "estimated_cost_usd": 0.01,
                        "throughput_tasks_per_sec": 0.08,
                        "token_usage_breakdown": {
                            "prompt_tokens": 60,
                            "completion_tokens": 40,
                            "total_tokens": 100,
                        },
                        "latency_percentiles": {
                            "p50": 12.4,
                            "p95": 12.4,
                        },
                    },
                    "hardware": {
                        "before": {"cpu_percent": 0.0},
                        "during": [{"cpu_percent": 12.0}],
                    },
                    "server_log_offsets": {
                        "vllm": {"start_byte": 1, "end_byte": 2},
                    },
                },
            )
            write_text(raw_fidelity / f"{sample_name}.log", "sample log body")

            summary = normalize_autogpt_dataset(root)

            self.assertEqual(summary["input_fidelity_dirs"], 1)
            self.assertEqual(summary["input_json_files"], 1)
            self.assertEqual(summary["output_csv_files"], 1)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["linked_log_files"], 1)
            self.assertEqual(summary["linked_hardware_files"], 1)

            csv_path = root / "simple-req1-code_generation" / "simple-req1-code_generation.csv"
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["ID"], "1")
            self.assertEqual(rows[0]["cfg-agent_config_id"], "agent_config_1")
            self.assertEqual(rows[0]["cfg-cycle_budget"], "3")
            self.assertEqual(rows[0]["cfg-ai-model_name"], "qwen2.5-7b-instruct")
            self.assertEqual(rows[0]["cfg-ai-temperature"], "0.7")
            self.assertEqual(rows[0]["obj-success_rate+"], "1.0")
            self.assertEqual(rows[0]["obj-avg_task_duration-"], "12.4")
            self.assertEqual(rows[0]["cost-total_tokens"], "100")
            self.assertEqual(rows[0]["cost-token_usage_prompt_tokens"], "60")
            self.assertEqual(rows[0]["hw-file"], "hw_file/hw-1.txt")
            self.assertEqual(rows[0]["log-file"], "log_file/log-1.txt")
            self.assertNotIn("FIDELITY_task_type", rows[0])

            hw_text = (csv_path.parent / rows[0]["hw-file"]).read_text(encoding="utf-8")
            self.assertIn('"cpu_percent": 12.0', hw_text)
            log_text = (csv_path.parent / rows[0]["log-file"]).read_text(encoding="utf-8")
            self.assertIn("===== SAMPLE LOG =====", log_text)
            self.assertIn("sample log body", log_text)
            self.assertIn("===== SERVER LOG OFFSETS =====", log_text)


if __name__ == "__main__":
    unittest.main()
