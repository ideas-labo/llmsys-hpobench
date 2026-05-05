# vLLM Performance Evaluation Manual

## vLLM Configurable Parameters

The following table lists the key vLLM parameters that can be adjusted to optimize performance. These parameters control various aspects of scheduling, memory management, and parallel processing.

| Non-AI Configuration Parameter | Type    | Range         | Description                                                          |
| ------------------------------ | ------- | ------------- | -------------------------------------------------------------------- |
| `tensor-parallel`              | Integer | [1, #GPUs]    | Number of tensor-parallel partitions across GPUs.                    |
| `max-num-seqs`                 | Integer | [64, 8192]    | Maximum number of concurrent sequences in the scheduler queue.       |
| `max-num-batched-tokens`       | Integer | [64, 8192]    | Maximum total number of tokens processed in a single batch.          |
| `block-size`                   | Enum    | {8, 16, 32}   | Size of memory blocks allocated for token storage.                   |
| `scheduler-delay-factor`       | Float   | [0, 2]        | Factor controlling batching delay before scheduling.                 |
| `enable-chunked-prefill`       | Boolean | {True, False} | Enables prefill chunking for reduced memory pressure on long inputs. |
| `enable-prefix-caching`        | Boolean | {True, False} | Enables caching of shared prefix tokens across requests.             |
| `disable-custom-all-reduce`    | Boolean | {True, False} | Disables optimized all-reduce communication.                         |
| `use-v2-block-manager`         | Boolean | {True, False} | Enables improved memory block management.                            |

---

# vLLM AI Component Parameters

These parameters have the most significant impact on **model performance** and **generation behavior** in vLLM.  
They are grouped into **Model Parameters**, **Speculative Decoding Parameters**, and **Sampling Parameters**.

---

## Model Parameters

| Parameter     | Type          | Default      | Description                                                                 | Recommended Range |
|---------------|--------------|--------------|-----------------------------------------------------------------------------|------------------|
| `model`       | `str`        | **Required** | Path or Hugging Face Hub ID of the main model.                              | e.g. `"Llama-3.1-8B"` |
---

## Speculative Decoding Parameters

| Parameter                                | Type    | Default | Description                                                                 | Recommended Range |
|------------------------------------------|---------|---------|-----------------------------------------------------------------------------|------------------|
| `model`                | `str`   | `None`  | Draft model name/path. Must share the exact same tokenizer with main model. | Smaller variant of the main model (e.g. 1B vs 4B) |
| `num_speculative_tokens` | `int` | `5`     | Number of tokens predicted by the draft model in each step.                 | `3–10` |
---

## Sampling Parameters (`SamplingParams`)

| Parameter             | Type    | Default | Description                                                               | Recommended Range |
|-----------------------|---------|---------|---------------------------------------------------------------------------|------------------|
| `top_k`               | `int`   | `-1`    | Limit sampling to top-k tokens. Lower = more deterministic.               | `10–50` (precise), `80–100` (creative) |
| `min_p`               | `float` | `0.0`   | Keep tokens until cumulative probability ≥ `min_p`. Increases diversity.  | `0.8–0.95` |
| `repetition_penalty`  | `float` | `1.0`   | Penalize repeated tokens from prompt/output. Reduces duplication.         | `1.1–1.3` |
| `length_penalty`      | `float` | `1.0`   | Bias toward shorter (`<1`) or longer (`>1`) completions.                  | `0.8–1.2` |
| `best_of`             | `int \| None` | `None` | Generate multiple outputs, return the highest-likelihood one.             | `2–5` |

---


## Environment / Fidelity Factors

The current organized vLLM dataset uses workload-level fidelity factors encoded in each fidelity directory and CSV file name:

```text
{request_rate}-{burstiness}-{max_concurrency}-{num_prompts}-r{repeat}
```

For example, `10.0-0.5-16-100-r1` means:

- `request_rate = 10.0`
- `burstiness = 0.5`
- `max_concurrency = 16`
- `num_prompts = 100`
- `repeat = 1`

These values are not duplicated as CSV columns. The fidelity directory and CSV file name are the source of truth for the workload fidelity point.

The current fidelity grid is:

| Fidelity Factor | Description | Current Values |
|:----------------|:------------|:---------------|
| `rate` | Target request submission rate used by the benchmark client. | `[5.0, 10.0, 15.0]` |
| `burstiness` | Shape parameter for Gamma-distributed request inter-arrival times. `1.0` approximates Poisson arrivals, smaller values are more bursty, and larger values are more regular. | `[0.5, 1.0, 2.0]` |
| `max_concurrency` | Maximum number of in-flight client requests. | `[4, 8, 16, 32]` |
| `num_prompts` | Number of prompts sampled from the ShareGPT workload for one benchmark run. This acts as the current workload-size fidelity axis. | `[50, 100, 200]` |
| `repeat` | Repeated run index for the same workload fidelity point. | `[1, 2]` |

This gives `3 x 3 x 4 x 3 x 2 = 216` fidelity points.

Notes:

- The current dataset uses `num_prompts` rather than a dataset subset ratio.
- Shared-prefix ratio is not a separate fidelity axis in the current organized data. Prefix-caching behavior appears as an AI/system configuration parameter through `cfg-ai-enable_prefix_caching`.
- Prompt complexity is fixed by the ShareGPT workload and is not encoded as a fidelity axis.

---
In the evaluation of original paper, authors synthesize workloads based on two distinct datasets to capture a variety of real-world scenarios:

### ShareGPT Dataset

- **Characteristics:**
  - The ShareGPT dataset is a collection of user-shared conversations with ChatGPT.
  - It features significantly longer input prompts (approximately 8.4× longer) and outputs (about 5.8× longer) compared to other datasets, with high variance.
- **Implications for Evaluation:**
  - The long prompts and outputs place a greater demand on memory management (e.g., KV cache handling).
  - Evaluations with ShareGPT stress-test the system’s ability to handle complex, high-variance workloads under high concurrency.

TODO: Add more datasets, such as Sonnet, BurstGPT, etc.

## Performance Metrics

When evaluating vLLM, focus on the following key performance metrics:

- **Throughput (Tokens/sec):** The number of tokens processed per second, which reflects the overall efficiency of the inference engine.
- **Time to First Token (TTFT):** The latency from when a request is received to when the first token is generated.
- **Token Processing Time (TPOT):** The average time taken to generate each subsequent token.
- **Normalized Latency:** The end-to-end latency of each request divided by the output length, which helps compare performance across requests with varying lengths.
- **Batching Efficiency:** The ratio of the actual batch size to the maximum possible batch size, reflecting scheduler effectiveness.

---

### Output Quality Metrics
- - **Successful Requests Ratio:**  

- **BLEU (Bilingual Evaluation Understudy):**  
  Measures N-gram precision against reference text with a brevity penalty.  

---

---

## Cost / Runtime Considerations

When evaluating vLLM performance, we focus on three key metrics that directly reflect resource consumption and runtime efficiency:

### 1. **gpu_cache_usage_perc_stats**  
- **Definition:** Percentage of GPU memory occupied by vLLM’s KV cache or internal buffers throughout the benchmark run.  
- **Why it matters:**  
  - Indicates memory pressure; sustained high utilization (e.g., > 90%) may risk OOM or reduced batch capacity.  
  - Helps tune parameters like `block-size`, `max-num-batched-tokens`, or enable `chunked-prefill` for efficiency.  
- **Collection method:** Exposed via the `vllm:gpu_cache_usage_perc` metric (0.85 means 85% usage) :contentReference[oaicite:1]{index=1}.  

### 2. **process_cpu_seconds_stats**  
- **Definition:** Total CPU time consumed by the vLLM process (sum of user + system CPU seconds).  
- **Why it matters:**  
  - Reflects CPU overhead for tokenization, scheduling, memory management, logging, etc.  
  - High CPU consumption may cause scheduling bottlenecks even if GPU remains underutilized.  
- **Collection method:** Use Prometheus `process_cpu_seconds_total`, which aggregates utime+stime for the process :contentReference[oaicite:2]{index=2}.

### 3. **benchmark_duration_s**  
- **Definition:** Wall-clock time (in seconds) from dispatch of the first request to receipt of the last response in the benchmark.  
- **Why it matters:**  
  - Captures end-to-end system efficiency, including all internal overheads (e.g., queueing, scheduling, GPU execution).  
  - Essential for evaluating throughput and latency under realistic load conditions.

---

### Summary Table

| Metric                         | Unit     | Significance                                                                 |
|-------------------------------|----------|------------------------------------------------------------------------------|
| `gpu_cache_usage_perc_stats`  | %        | Memory pressure and capacity for batching                                   |
| `process_cpu_seconds_stats`   | seconds  | CPU overhead from scheduling, tokenization, and memory management           |
| `benchmark_duration_s`        | seconds  | End-to-end runtime reflecting overall system efficiency and throughput      |

---
