# Case Study Reproduction

This directory records the recovered case-study protocol used for the published offline comparisons. The executable implementation lives in `llmsys/` at the repository root.

## Environment

Use the locked Python 3.10 environment:

```bash
uv sync --extra case-study --extra dev --frozen
```

The archived dependency versions are:

- NumPy 1.24.4
- pandas 2.0.3
- SciPy 1.10.1
- scikit-learn 1.6.1
- PyYAML 6.0.2
- ConfigSpace 1.2.2
- SMAC 2.4.0
- HEBO 0.3.6

## Data layout

Place the normalized case-study tables at:

```text
datasets/<system>/<fidelity>/<fidelity>.csv
```

The configuration file names the exact fidelity table, objective label, and cost label for each system. The recovered Table 2 loader treats every `cfg-*` column as optimizer-visible, including any provenance identifier that had already been normalized with that prefix. This behavior is preserved here because it affects the historical search space. The later coverage-corrected source excludes `id`, `config_id`, `agent_config_id`, `repeat`, and `experiment_id`; its source hash and configuration are recorded separately in `data_manifest.json`.

`experiment_table2.yaml` is the recovered seven-system Table 2 configuration. Its density generator takes seeded, nested prefixes from the exact cross-fidelity configuration intersection. `experiment.yaml` records the later four-system coverage-corrected configuration and its separately hashed density manifest. Generate the archived 20%, 40%, 60%, and 80% subsets with seed `20260731`:

```bash
uv run python -m llmsys.prepare_densities --config experiments/case_study/experiment_table2.yaml
```

Verify the generated `datasets_processed_coverage_fixed/manifest.json` SHA-256 against `data_manifest.json`.

## Evaluator behavior

For the requested fidelity and label, the evaluator averages finite values from duplicate rows with an exact complete-configuration match. A missing exact objective and a missing exact cost trigger separate fallback predictions.

The `rf` fallback trains one model per fidelity and label. It uses 200 trees, the run-derived random seed, one worker, typed categorical tokens, `OneHotEncoder(handle_unknown="ignore")`, and unscaled numeric features. The `1nn` fallback divides every numeric difference by that parameter's observed full-data range and counts each categorical mismatch as one.

The evaluator accepts a new query while cumulative returned cost remains below the budget. It then adds the complete returned cost, including a final query that crosses the budget boundary.

## Commands

Validate and run one job:

```bash
uv run --extra case-study python experiments/case_study/run.py \
  --config experiments/case_study/experiment_table2.yaml \
  --fallback rf --density 100 --seed 0 --system vLLM \
  --tuner random_search --workers 1
```

Run the configured job matrix:

```bash
uv run --extra case-study python experiments/case_study/run.py \
  --config experiments/case_study/experiment_table2.yaml
```

Result files use this path:

```text
results_3x/<fallback>/<density>/run<seed>/<system>/<tuner>/results.csv
```

Every result row contains the complete optimizer configuration, objective, cost, and fidelity name. The selected YAML file is the authoritative specification for that recovered protocol snapshot.
