import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_croissant import generate_croissant


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


class GenerateCroissantTests(unittest.TestCase):
    def test_generates_manifest_and_croissant_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "experiment-data"
            fidelity_dir = data_root / "Engine" / "vLLM" / "5.0-0.5-4-50-r1"
            write_text(
                fidelity_dir / "5.0-0.5-4-50-r1.csv",
                """
ID,cfg-config_id,obj-score+,hw-file,log-file
1,465,0.9,hw_file/hw-1.txt,log_file/log-1.txt
2,466,0.8,,log_file/log-2.txt
""",
            )
            write_text(fidelity_dir / "hw_file" / "hw-1.txt", "gpu")
            write_text(fidelity_dir / "log_file" / "log-1.txt", "log one")
            write_text(fidelity_dir / "log_file" / "log-2.txt", "log two")

            output = root / "croissant.json"
            records_output = root / "metadata" / "croissant_records.csv"
            summary = generate_croissant(
                data_root=data_root,
                output_path=output,
                records_output_path=records_output,
                dataset_url="https://example.org/llmsys-hpobench",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                creators=["LLMSYS-HPOBench Authors"],
            )

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["systems"], 1)
            self.assertEqual(summary["fidelities"], 1)
            self.assertEqual(summary["missing_artifacts"], 0)
            self.assertTrue(output.is_file())
            self.assertTrue(records_output.is_file())

            with records_output.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["system"], "vLLM")
            self.assertEqual(rows[0]["fidelity"], "5.0-0.5-4-50-r1")
            self.assertEqual(rows[0]["record_id"], "1")
            self.assertEqual(rows[0]["csv_file"], "Engine/vLLM/5.0-0.5-4-50-r1/5.0-0.5-4-50-r1.csv")
            self.assertEqual(rows[0]["hw_file"], "Engine/vLLM/5.0-0.5-4-50-r1/hw_file/hw-1.txt")
            self.assertEqual(rows[0]["log_file"], "Engine/vLLM/5.0-0.5-4-50-r1/log_file/log-1.txt")

            metadata = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(metadata["@type"], "sc:Dataset")
            self.assertEqual(metadata["name"], "LLMSYS-HPOBench")
            self.assertIn("Anonymous", metadata["citeAs"])
            self.assertIn(
                "LLMSYS-HPOBench: Hyperparameter Optimization Benchmark Suite for Real-World LLM Systems",
                metadata["citeAs"],
            )
            self.assertIn("rai:dataUseCases", metadata)
            self.assertIn("rai:dataLimitations", metadata)
            self.assertIsInstance(metadata["rai:dataLimitations"], list)
            self.assertIn("rai:dataBiases", metadata)
            self.assertIn("rai:personalSensitiveInformation", metadata)
            self.assertIn("rai:dataSocialImpact", metadata)
            self.assertIs(metadata["rai:hasSyntheticData"], True)
            self.assertIn("rai:sourceDatasets", metadata)
            self.assertIn(
                "ShareGPT Vicuna unfiltered conversation dataset: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
                metadata["rai:sourceDatasets"],
            )
            self.assertIn(
                "SGLang generated shared-prefix workload recipe and seeds: https://zenodo.org/records/20048594",
                metadata["rai:sourceDatasets"],
            )
            self.assertIn("rai:provenanceActivities", metadata)
            provenance_names = {activity["name"] for activity in metadata["rai:provenanceActivities"]}
            self.assertIn("Synthetic shared-prefix workload generation", provenance_names)
            self.assertIn("prov:wasDerivedFrom", metadata)
            self.assertIn(
                "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
                metadata["prov:wasDerivedFrom"],
            )
            self.assertIn("prov:wasGeneratedBy", metadata)
            generated_by_ids = {activity["@id"] for activity in metadata["prov:wasGeneratedBy"]}
            self.assertIn("activity_raw_collection", generated_by_ids)
            self.assertIn("activity_synthetic_shared_prefix_generation", generated_by_ids)
            self.assertIn("activity_croissant_generation", generated_by_ids)
            self.assertIn("prov", metadata["@context"])
            self.assertIn("distribution", metadata)
            self.assertIn("recordSet", metadata)
            self.assertEqual(metadata["recordSet"][0]["@id"], "sample_manifest")
            field_names = [field["name"] for field in metadata["recordSet"][0]["field"]]
            self.assertIn("system", field_names)
            self.assertIn("log_file", field_names)


if __name__ == "__main__":
    unittest.main()
