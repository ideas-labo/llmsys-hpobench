import unittest
import tempfile
from pathlib import Path

import pandas as pd

from llmsys.data import (
    DatasetBundle,
    ExperimentConfig,
    FidelityTable,
    ParameterSpec,
    SearchSpace,
    SystemSpec,
)
from llmsys.evaluator import OfflineEvaluator
from llmsys.fallbacks import NearestNeighborFallback, RandomForestFallback
from llmsys.runner import Job, run_job


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".codex-work" / "tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class CaseStudyEvaluatorTests(unittest.TestCase):
    def make_bundle(self) -> DatasetBundle:
        space = SearchSpace(
            (
                ParameterSpec("cfg-size", "integer", lower=1, upper=100),
                ParameterSpec("cfg-mode", "categorical", choices=("a", "b")),
            )
        )
        frame = pd.DataFrame(
            [
                {"cfg-size": 1, "cfg-mode": "a", "obj-score+": 1.0, "cost-runtime": 4.0},
                {"cfg-size": 1, "cfg-mode": "a", "obj-score+": 3.0, "cost-runtime": 6.0},
                {"cfg-size": 100, "cfg-mode": "b", "obj-score+": 9.0, "cost-runtime": 8.0},
            ]
        )
        table = FidelityTable(
            name="f1",
            frame=frame,
            config_columns=("cfg-size", "cfg-mode"),
            space=space,
        )
        spec = SystemSpec(
            name="system",
            fidelities={1: "f1"},
            objective="obj-score+",
            cost="cost-runtime",
        )
        return DatasetBundle(
            spec=spec,
            tables={1: table},
            config_columns=("cfg-size", "cfg-mode"),
            space=space,
        )

    def test_exact_duplicate_values_are_averaged_without_fallback_models(self):
        bundle = self.make_bundle()
        fallback = RandomForestFallback(bundle.space, seed=13)
        evaluator = OfflineEvaluator(bundle, fallback, budget_seconds=100)

        result = evaluator.evaluate({"cfg-size": 1, "cfg-mode": "a"}, 1)

        self.assertEqual(result.objective, 2.0)
        self.assertEqual(result.cost, 5.0)
        self.assertEqual(fallback.model_count, 0)

    def test_random_forest_trains_separate_objective_and_cost_models(self):
        bundle = self.make_bundle()
        fallback = RandomForestFallback(bundle.space, seed=13)
        evaluator = OfflineEvaluator(bundle, fallback, budget_seconds=100)

        result = evaluator.evaluate({"cfg-size": 50, "cfg-mode": "new"}, 1)

        self.assertIsInstance(result.objective, float)
        self.assertGreater(result.cost, 0)
        self.assertEqual(fallback.model_count, 2)

    def test_budget_charges_the_complete_crossing_query(self):
        bundle = self.make_bundle()
        evaluator = OfflineEvaluator(
            bundle,
            NearestNeighborFallback(bundle.space),
            budget_seconds=3,
        )

        result = evaluator.evaluate({"cfg-size": 1, "cfg-mode": "a"}, 1)

        self.assertEqual(result.cumulative_cost, 5.0)
        self.assertTrue(evaluator.exhausted)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            evaluator.evaluate({"cfg-size": 100, "cfg-mode": "b"}, 1)

    def test_nearest_neighbor_normalizes_numeric_parameter_ranges(self):
        bundle = self.make_bundle()
        fallback = NearestNeighborFallback(bundle.space)
        table = bundle.tables[1]

        prediction = fallback.predict(
            table,
            "obj-score+",
            {"cfg-size": 90, "cfg-mode": "a"},
        )

        self.assertEqual(prediction, 1.0)

    def test_runner_writes_a_complete_random_search_ledger(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "datasets"
            for rung, fidelity in ((1, "f1"), (2, "f2"), (4, "f4")):
                path = dataset_root / "system" / fidelity / f"{fidelity}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "cfg-size,obj-score+,cost-runtime\n"
                    f"1,{rung}.0,1.0\n"
                    f"10,{rung + 1}.0,2.0\n",
                    encoding="utf-8",
                )
            spec = SystemSpec(
                name="system",
                fidelities={1: "f1", 2: "f2", 4: "f4"},
                objective="obj-score+",
                cost="cost-runtime",
            )
            config = ExperimentConfig(
                dataset_root=dataset_root,
                processed_root=root / "processed",
                results_root=root / "results",
                densities=(100,),
                fallbacks=("rf",),
                tuners=("random_search",),
                seeds=(0,),
                budget_seconds=2.5,
                workers=1,
                density_seed=20260731,
                systems={"system": spec},
            )

            outputs = run_job(
                config,
                Job(fallback="rf", density=100, seed=0, system="system"),
                tuners=("random_search",),
            )

            self.assertEqual(len(outputs), 1)
            result = pd.read_csv(outputs[0])
            self.assertGreaterEqual(result["cost-runtime"].sum(), 2.5)
            self.assertEqual(
                list(result.columns),
                ["cfg-size", "obj-score+", "cost-runtime", "fidelity"],
            )


if __name__ == "__main__":
    unittest.main()
