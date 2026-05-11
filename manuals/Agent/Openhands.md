AI
| Parameter                   | Type        | Approx. value range (incl. default)       | Short description (performance impact)                                   |
| --------------------------- | ----------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| `model`                     | `str`       | Any model name; default per docs          | Biggest driver of latency/cost/quality                                   |
| `max_message_chars`         | `int`       | ~1,000–200,000 (common); default `30000`  | Larger = more context but slower/more expensive; truncates when exceeded |
| `max_input_tokens`          | `int`       | ~0–200,000 (model-dependent); default `0` | Caps input size; lower saves cost/time but can drop info                 |
| `max_output_tokens`         | `int`       | ~0–32,000 (model-dependent); default `0`  | Caps output length; lower saves cost/time but can truncate answers       |
| `caching_prompt`            | `bool`      | `true/false`; default `true`              | If supported, reduces repeated-call latency/cost                         |
| `temperature`               | `float`     | ~0.0–2.0 (common); default `0.0`          | Higher = more randomness; can increase wasted tokens/redo                |
| `top_p`                     | `float`     | **0.0–1.0** (standard); default `1.0`     | Lower = more conservative/stable (often more efficient)                  |
| `timeout` (LLM)             | `int`       | ~0–600s (common); default `0`             | Too low → timeouts/retries; too high → long hangs                        |
| `num_retries`               | `int`       | ~0–10 (common); default `8`               | More retries = more robust but slower on failures                        |
| `retry_min_wait`            | `int`       | ~0–60s (common); default `15`             | Minimum backoff between retries; higher slows failure recovery           |
| `retry_max_wait`            | `int`       | ~0–600s (common); default `120`           | Maximum backoff; higher can greatly extend failure time                  |
| `retry_multiplier`          | `float`     | ~1.0–3.0 (common); default `2.0`          | Exponential backoff growth rate                                          |
| `disable_vision`            | `bool/None` | `true/false/unset`; default `None`        | Disabling vision saves cost/time when images aren’t needed               |
| `function_calling`          | `bool`      | `true/false`; default `true`              | Enables tools; adds overhead but often required for workflows            |
| `enable_browsing`           | `bool`      | `true/false`; default `false`             | Web browsing usually adds major latency/cost                             |
| `enable_history_truncation` | `bool`      | `true/false`; default `true`              | Improves run completion under context limits (stability)                 |


Non-AI
| Parameter              | Type    | Approx. value range (incl. default)                      | Short description (performance impact)                      |
| ---------------------- | ------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| `max_iterations`       | `int`   | ~1–10,000 (common); default `100`                        | More iterations generally = more time/cost                  |
| `max_budget_per_task`  | `float` | **≥0**; default `0.0` (`0.0` = unlimited)                | Hard cap on task spend/time; key cost-control lever         |
| `runtime`              | `str`   | Unspecified; default `docker`                            | Runtime choice affects startup/execution characteristics    |
| `volumes`              | `str`   | `host:container[:mode]`, comma-separated; default `None` | Mount/I/O strategy can materially affect performance        |
| `timeout` (sandbox)    | `int`   | ~1–3600s (common); default `120`                         | Too low → frequent timeouts; too high → long stalls         |
| `base_container_image` | `str`   | Any image name; default per docs                         | Image size & preinstalled deps affect startup and execution |
| `use_host_network`     | `bool`  | `true/false`; default `false`                            | Can affect network throughput/latency and connectivity      |
| `enable_auto_lint`     | `bool`  | `true/false`; default `false`                            | Adds extra time after edits; usually off for speed          |
| `runtime_extra_deps`   | `str`   | Unspecified; default `""`                                | Installing extra deps increases setup/build time            |

## Fidelity
Dataset1
| Fidelity Factor       | Description                                                                                                            | Configuration Range    |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| **Context Sentences** | Number of sentences (`sent1…sentN`) in the knowledge context. Measures retrieval load and context noise.               | `[10, 20, 40, 80]`     |
| **Facts Count**       | Number of atomic fact statements (non-rule sentences). Tests filtering and evidence selection.                         | `[5, 10, 20, 40]`      |
| **Rules Count**       | Number of logical rule statements ("If … then …"). Controls reasoning search space size.                               | `[2, 4, 8, 16]`        |
| **Proof Depth**       | Required multi-hop reasoning steps inferred from proof chain length. Measures long-chain reasoning and state tracking. | `[1, 2, 3, 5, 7]`      |
| **Distractor Ratio**  | Ratio of irrelevant sentences to total context. Measures robustness to noise.                                          | `[0, 0.25, 0.5, 0.75]` |

The current normalized ProofWriter/OpenHands samples use hyphen-separated fidelity values:

```text
{facts_count}-{rules_count}-{proof_depth}-{context_sentences}
```

Example: `7-7-1-1`.


### Cost / Runtime Considerations
- **Tokens:** Additional time due to function parsing, execution, and queue updates.
- **CPU time:** the cumulative amount of time during which the CPU is actively executing the process, including both user-mode and system-mode CPU time.
- **Peak Memory Usage:** the maximum amount of memory consumed by the process during execution, typically measured as the peak resident set size.

### Performance
(DATASET1)
- **Accuracy:** Fraction of runs whose final answer matches the gold answer.
- **Evidence Recall:** Measures how much of the gold supporting evidence is actually involved in the model’s reasoning process, i.e., the participation rate of correct evidence.
- **Proof Step Deviation:** Absolute difference between the number of proof steps generated by the model and the number of steps in the gold proof; this metric compares only the step count, not step correctness.
- **Total Time:** time used from start.


