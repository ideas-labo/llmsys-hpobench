import math
import tempfile
import unittest
from pathlib import Path

from llmsys_hpobench import Benchmark


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".codex-work" / "tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


class ExperimentalEvaluatorTests(unittest.TestCase):
    def test_exact_values_are_averaged_per_metric_without_fitting_models(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            write_csv(
                root / "system" / "f1" / "f1.csv",
                """
ID,cfg-size,cfg-mode,obj-score+,cost-runtime,log-file
1,1,a,1.0,10.0,log_file/log-1.txt
2,1,a,3.0,14.0,log_file/log-2.txt
3,2,b,5.0,20.0,log_file/log-3.txt
""",
            )
            benchmark = Benchmark("system", root=root, random_state=7)

            measurement = benchmark.evaluate({"size": 1, "mode": "a"}, "f1")

            self.assertEqual(measurement["perf"]["score+"], 2.0)
            self.assertEqual(measurement["cost"]["runtime"], 12.0)
            self.assertEqual(measurement["source"]["perf"]["score+"], "exact")
            self.assertEqual(measurement["source"]["cost"]["runtime"], "exact")
            self.assertEqual(benchmark._experimental_evaluator.model_count, 0)

    def test_missing_metric_uses_a_separate_random_forest_for_each_label(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            write_csv(
                root / "system" / "f1" / "f1.csv",
                """
ID,cfg-size,cfg-mode,obj-score+,obj-quality+,cost-runtime
1,1,a,1.0,10.0,5.0
2,2,a,2.0,20.0,6.0
3,3,b,3.0,30.0,7.0
""",
            )
            benchmark = Benchmark("system", root=root, random_state=11)

            measurement = benchmark.evaluate({"size": 4, "mode": "new"}, "f1")

            self.assertEqual(
                measurement["source"],
                {
                    "perf": {
                        "score+": "random_forest",
                        "quality+": "random_forest",
                    },
                    "cost": {"runtime": "random_forest"},
                },
            )
            self.assertTrue(math.isfinite(measurement["perf"]["score+"]))
            self.assertTrue(math.isfinite(measurement["perf"]["quality+"]))
            self.assertTrue(math.isfinite(measurement["cost"]["runtime"]))
            self.assertEqual(benchmark._experimental_evaluator.model_count, 3)
            self.assertEqual(measurement["hardware"], {})
            self.assertEqual(measurement["log"], {})
            self.assertEqual(measurement["row"], {})

    def test_non_finite_exact_value_uses_random_forest_only_for_that_metric(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            write_csv(
                root / "system" / "f1" / "f1.csv",
                """
ID,cfg-size,obj-score+,cost-runtime
1,1,nan,4.0
2,2,2.0,5.0
3,3,3.0,6.0
""",
            )
            benchmark = Benchmark("system", root=root, random_state=3)

            measurement = benchmark.evaluate({"size": 1}, "f1")

            self.assertEqual(measurement["source"]["perf"]["score+"], "random_forest")
            self.assertEqual(measurement["source"]["cost"]["runtime"], "exact")
            self.assertEqual(measurement["cost"]["runtime"], 4.0)
            self.assertEqual(benchmark._experimental_evaluator.model_count, 1)

    def test_models_are_cached_separately_by_fidelity(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            for fidelity, offset in (("f1", 0), ("f2", 10)):
                write_csv(
                    root / "system" / fidelity / f"{fidelity}.csv",
                    f"""
ID,cfg-size,obj-score+,cost-runtime
1,1,{1 + offset}.0,{5 + offset}.0
2,2,{2 + offset}.0,{6 + offset}.0
""",
                )
            benchmark = Benchmark("system", root=root, random_state=5)

            benchmark.evaluate({"size": 3}, "f1")
            benchmark.evaluate({"size": 3}, "f2")

            self.assertEqual(benchmark._experimental_evaluator.model_count, 4)

    def test_random_state_makes_predictions_reproducible(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            write_csv(
                root / "system" / "f1" / "f1.csv",
                """
ID,cfg-size,cfg-mode,obj-score+,cost-runtime
1,1,a,1.0,5.0
2,2,a,3.0,6.0
3,3,b,2.0,7.0
4,4,b,8.0,8.0
""",
            )
            left = Benchmark("system", root=root, random_state=17)
            right = Benchmark("system", root=root, random_state=17)

            left_result = left.evaluate({"size": 5, "mode": "c"}, "f1")
            right_result = right.evaluate({"size": 5, "mode": "c"}, "f1")

            self.assertEqual(left_result["perf"], right_result["perf"])
            self.assertEqual(left_result["cost"], right_result["cost"])

    def test_incomplete_or_unknown_configuration_fails_immediately(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            write_csv(
                root / "system" / "f1" / "f1.csv",
                """
ID,cfg-size,cfg-mode,obj-score+
1,1,a,1.0
""",
            )
            benchmark = Benchmark("system", root=root)

            with self.assertRaisesRegex(ValueError, "missing=.*mode"):
                benchmark.evaluate({"size": 1}, "f1")
            with self.assertRaisesRegex(ValueError, "unknown=.*extra"):
                benchmark.evaluate({"size": 1, "mode": "a", "extra": 2}, "f1")


if __name__ == "__main__":
    unittest.main()
