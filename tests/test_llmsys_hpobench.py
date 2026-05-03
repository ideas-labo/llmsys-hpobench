import tempfile
import unittest
from pathlib import Path

from llmsys_hpobench import Benchmark, register_system, registered_systems


def write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_exposes_spaces_and_evaluates_prefixed_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            write_csv(
                tmp_path
                / "target_system"
                / "moderate-r1-memory_retrieval"
                / "moderate-r1-memory_retrieval.csv",
                """
ID,cfg-max_num_seqs,cfg-enable_prefix_caching,cfg-ai-temperature,FIDELITY_factor,FIDELITY_repeat,obj-throughput+,obj-TTFT-,cost-gpu_cache_usage,hw-file,log-client-file,log-server-file
1,1024,True,0.2,moderate,1,145.2,0.62,84.7,hw_file/id1-hw.csv,log_file/id1-client.log,log_file/id1-server.log
2,2048,False,0.7,moderate,1,167.9,0.71,90.8,hw_file/id2-hw.csv,log_file/id2-client.log,log_file/id2-server.log
""",
            )
            write_csv(
                tmp_path
                / "target_system"
                / "moderate-r1-memory_retrieval"
                / "hw_file"
                / "id2-hw.csv",
                """
timestamp,gpu_util
1,80
""",
            )

            benchmark = Benchmark(system="target_system", root=tmp_path)

            X = benchmark.get_config_space()
            Z = benchmark.get_fidelity_space()

            self.assertEqual(
                X.columns,
                [
                    "max_num_seqs",
                    "enable_prefix_caching",
                    "temperature",
                ],
            )
            self.assertEqual(X.ai_columns, ["temperature"])
            self.assertEqual(X.non_ai_columns, ["max_num_seqs", "enable_prefix_caching"])
            self.assertEqual(Z.names, ["moderate-r1-memory_retrieval"])

            z = Z.sample(random_state=0)
            x = X.sample(fidelity=z, random_state=0)
            measurement = benchmark.evaluate(config=x, fidelity=z)

            self.assertEqual(measurement["perf"]["throughput+"], 167.9)
            self.assertEqual(measurement["perf"]["TTFT-"], 0.71)
            self.assertEqual(measurement["cost"]["gpu_cache_usage"], 90.8)
            self.assertEqual(measurement["hardware"]["file"], "hw_file/id2-hw.csv")
            self.assertEqual(measurement["log"]["client-file"], "log_file/id2-client.log")
            self.assertEqual(measurement["log"]["server-file"], "log_file/id2-server.log")
            self.assertEqual(
                measurement.select(perf=["TTFT-"], cost=["gpu_cache_usage"], log=["client-file"]),
                {
                    "perf": {"TTFT-": 0.71},
                    "cost": {"gpu_cache_usage": 90.8},
                    "log": {"client-file": "log_file/id2-client.log"},
                },
            )
            self.assertEqual(Z.names, ["moderate-r1-memory_retrieval"])
            self.assertEqual(len(benchmark.records), 2)

    def test_unprefixed_csv_is_rejected_until_dataset_is_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            write_csv(
                tmp_path / "vLLM" / "rate10.0_burst0.5_conc16_prompts100.csv",
                """
id,config_id,repeat,tp_size,max_num_seqs,enable_prefix_caching,temperature,benchmark_duration_s,request_throughput,mean_ttft_ms,successful_requests,failed_requests,gpu_kv_cache_usage_avg,process_cpu_seconds_avg
1,465,1,1,1029,True,0.154,23.47,3.93,29.17,92,8,0.057,170.57
""",
            )

            with self.assertRaisesRegex(ValueError, "No normalized benchmark columns"):
                Benchmark(system="vLLM", root=tmp_path)

    def test_missing_exact_config_raises_when_nearest_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            write_csv(
                tmp_path / "target_system" / "f1" / "f1.csv",
                """
ID,cfg-a,obj-score+
1,1,0.5
""",
            )

            benchmark = Benchmark(system="target_system", root=tmp_path, on_missing="error")

            with self.assertRaises(KeyError):
                benchmark.evaluate({"a": 2}, "f1")

    def test_registered_system_resolves_experiment_data_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            write_csv(
                tmp_path / "experiment-data" / "Engine" / "vLLM" / "f1" / "f1.csv",
                """
ID,cfg-a,obj-score+,log-client-file,hw-file
1,1,0.5,log_file/id1-client.log,
""",
            )

            benchmark = Benchmark(system="vLLM", root=tmp_path / "experiment-data")

            self.assertEqual(benchmark.system_dir, (tmp_path / "experiment-data" / "Engine" / "vLLM").resolve())
            measurement = benchmark.evaluate({"a": 1}, "f1")
            self.assertEqual(measurement["perf"]["score+"], 0.5)
            self.assertEqual(measurement["log"]["client-file"], "log_file/id1-client.log")
            self.assertEqual(measurement["hardware"]["file"], None)

    def test_custom_system_registration_extends_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            write_csv(
                tmp_path / "experiment-data" / "Custom" / "my_system" / "f1" / "f1.csv",
                """
ID,cfg-a,obj-score+
1,1,0.5
""",
            )

            register_system("my_system", "Custom/my_system")
            self.assertEqual(registered_systems()["my_system"], "Custom/my_system")

            benchmark = Benchmark(system="my_system", root=tmp_path / "experiment-data")

            self.assertEqual(benchmark.get_fidelity_space().names, ["f1"])

    def test_unregistered_system_is_auto_discovered_when_unambiguous(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            write_csv(
                tmp_path / "experiment-data" / "NewCategory" / "new_system" / "f1" / "f1.csv",
                """
ID,cfg-a,obj-score+
1,1,0.5
""",
            )

            benchmark = Benchmark(system="new_system", root=tmp_path / "experiment-data")

            self.assertEqual(benchmark.system_dir, (tmp_path / "experiment-data" / "NewCategory" / "new_system").resolve())

    def test_unregistered_system_auto_discovery_rejects_ambiguous_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            for category in ["One", "Two"]:
                write_csv(
                    tmp_path / "experiment-data" / category / "duplicate" / "f1" / "f1.csv",
                    """
ID,cfg-a,obj-score+
1,1,0.5
""",
                )

            with self.assertRaisesRegex(ValueError, "Ambiguous system name"):
                Benchmark(system="duplicate", root=tmp_path / "experiment-data")


if __name__ == "__main__":
    unittest.main()
