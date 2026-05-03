# SGLang Performance Evaluation Manual

## Configurable Parameters

<!-- The following table lists the key SGLang parameters that can be adjusted to optimize performance. These parameters control various aspects of scheduling, memory management, and parallel processing. -->

<!-- | Non-AI Configuration Parameter | Type    | Range         | Description                                                          |
| ------------------------------ | ------- | ------------- | -------------------------------------------------------------------- |
| `tensor-parallel`              | Integer | [1, #GPUs]    | Number of tensor-parallel partitions across GPUs.                    | -->

This section highlights the **most performance-critical parameters** in SGLang for  **multi-fidelity sampling**. These parameters may impact **throughput, latency, memory usage, and accuracy**.

---

## 1. Model and Context

| Parameter            | Type     | Values                                | Description                                                                 |
|----------------------|----------|---------------------------------------|-----------------------------------------------------------------------------|
| `--model-path`       | String   | Path / HuggingFace repo ID            | Location of model weights.                                                  |
| `--context-length`   | Integer  | [1, model max]                        | Maximum input context length (affects memory + latency).                     |

---

## 2. Parallelism (Not Applicable for Our Device)

| Parameter   | Type    | Values                  | Description                                        |
|-------------|---------|-------------------------|----------------------------------------------------|
| `--tp-size` | Integer | [1, #GPUs]              | Tensor parallelism size (scales across GPUs).      |
| `--dp-size` | Integer | [1, #GPUs]              | Data parallelism size (improves throughput).       |
| `--pp-size` | Integer | [1, #GPUs]              | Pipeline parallelism size (useful for large models).|

---

## 3. Memory and Scheduling

| Parameter              | Type     | Values             | Description                                                                 |
|------------------------|----------|--------------------|-----------------------------------------------------------------------------|
| `--mem-fraction-static`| Float    | (0,1]              | Fraction of GPU memory for static allocation.                               |
| `--max-total-tokens`   | Integer  | ≥1                 | Total tokens allowed in memory pool.                                        |
| `--chunked-prefill-size`| Integer | ≥-1                | Chunk size for prefill (-1 disables).                                       |
| `--schedule-policy`    | Enum     | {fcfs, priority}   | Scheduling strategy (fairness vs throughput trade-off).                      |

---

## 4. Quantization and Precision

| Parameter          | Type   | Values                                  | Description                                                                 |
|--------------------|--------|-----------------------------------------|-----------------------------------------------------------------------------|
| `--dtype`          | Enum   | {auto, float16, bfloat16, float32}      | Precision for weights and activations.                                      |
| `--quantization`   | Enum   | {fp8, int8, int4, torchao variants}     | Weight quantization strategy.                                               |
| `--kv-cache-dtype` | Enum   | {auto, fp8_e5m2, fp8_e4m3}              | KV cache precision (affects speed vs accuracy).                             |

---

## 5. Speculative Decoding

| Parameter                         | Type     | Values        | Description                                                                 |
|----------------------------------|----------|---------------|-----------------------------------------------------------------------------|
| `--speculative-algorithm`        | String   | {eagle, draft}| Speculative decoding method.                                                |
| `--speculative-draft-model-path` | String   | Path / Repo   | Draft model used for speculative decoding.                                  |
| `--speculative-num-steps`        | Integer  | ≥1            | Number of draft decoding steps.                                             |

---

## 6. Logging and Metrics

| Parameter          | Type     | Values                  | Description                                                                 |
|--------------------|----------|-------------------------|-----------------------------------------------------------------------------|
| `--enable-metrics` | Boolean  | {True, False}           | Enable Prometheus metrics collection.                                       |
| `--log-level`      | Enum     | {debug, info, warning}  | Logging verbosity (affects runtime overhead).                               |

---

## Performance Factor Mapping

| Performance Factor  | Parameters                                                                 | Impact                                                                 |
|---------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------|
| **Scalability**     | `--tp-size`, `--dp-size`, `--pp-size`                                      | Determines how computation is distributed across GPUs.                  |
| **Memory**          | `--mem-fraction-static`, `--max-total-tokens`, `--max-running-requests`    | Controls GPU memory allocation between KV cache and activations.        |
| **Precision**       | `--dtype`, `--quantization`, `--kv-cache-dtype`                            | Balances accuracy vs. performance via quantization & lower precision.   |
| **Scheduling**      | `--schedule-policy`, `--schedule-conservativeness`, `--chunked-prefill-size`| Affects latency, concurrency, and cache efficiency.                     |
| **Speculative**     | `--speculative-*` (algorithm, draft model, num-steps)                      | Trades computation cost for faster decoding via draft models.           |
| **Optimization**    | `--cuda-graph-max-bs`, `--enable-torch-compile`, `--enable-dp-attention`   | Advanced optimizations for batch execution and parallelism.             |

---

## Sampling Paramters (`AI COMPONENT PARAMETERS`)

These parameters have the most significant impact on generation behavior in SGLang. Adjust them to control randomness, repetition, and output structure.

| Parameter             | Type            | Default | Description                                                               | Recommended Range             |
|-----------------------|------------------|---------|---------------------------------------------------------------------|-------------------------------|
| `top_k`               | `int`           | `-1`    | Limit sampling to top-k tokens. Lower = more deterministic.          | `10–50` (precise), `80–100` (creative) |
| `min_p`               | `float`         | `0.0`   | Keep tokens until cumulative prob ≥ `min_p`. Increases diversity.    | `0.8–0.95`                    |
| `repetition_penalty` | `float`         | `1.0`   | Penalize repeated tokens from prompt/output. Reduces duplication.     | `1.1–1.3`                     |
| `length_penalty`      | `float`         | `1.0`   | Bias toward shorter (`<1`) or longer (`>1`) completions.             | `0.8–1.2`                     |
| `best_of`             | `int \| None`   | `None`  | Generate multiple outputs, return the highest-likelihood one.         | `2–5`                        |


## Environment / Fidelity Factors

To ensure that evaluation results accurately reflect real-world deployments, several environmental and fidelity factors must be controlled. The table below lists the key factors considered in our SGLang evaluation:

| **Fidelity Factor**            | **Description**                                                                                              | **Configuration Range**                |
|:------------------------------|:-------------------------------------------------------------------------------------------------------------|:---------------------------------------|
| **Request Rate**               | Target request submission rate in requests per second. `inf` represents maximum throughput testing without rate limiting. | `[5.0, 15.0, inf]`                      |
| **Concurrency Level**          | Number of requests processed simultaneously, affecting batching and scheduler performance                     | `[16]` (fixed at medium level)                        |
| **Burstiness**    | Shape parameter of the Gamma distribution used to model request inter-arrival times. A value of `1.0` results in an exponential distribution (standard Poisson process with uniform arrivals). Higher values (e.g., `2.0`) produce more bursty traffic patterns with temporal clustering of requests. | `[1.0, 2.0]` |
| **Dataset Type**               | The dataset used for evaluation, which determines workload characteristics and cache behavior                                   | `{generated-shared-prefix}`  |

---

### Parameters for Generated Shared Prefix Dataset

When the `generated-shared-prefix` dataset is selected, the following parameters control the structure of shared prefixes and question-answer pairs. These parameters are critical for evaluating **Radix Cache** performance:

| **Parameter**                  | **Type**     | **Configuration Range** | **Description**                                                                 |
|----------------------------|----------|---------|-----------------------------------------------------------------------------|
| `gsp_num_groups`         | Integer  | `[32, 64, 128]`    | Number of unique system prompts (shared prefixes). Lower values result in higher cache hit rates, while higher values reduce prefix reuse and test cache capacity under diverse workloads.                                           |
| `gsp_prompts_per_group`  | Integer  | `[16]`    | Number of unique questions per system prompt (fixed). Each group reuses the same system prompt with different questions, simulating shared-context scenarios.                                               |
| `gsp_system_prompt_len`  | Integer  | `[2048]`  | Length of the system prompt (shared prefix) in tokens (fixed). Longer prefixes increase memory savings from caching but also increase initial prefill cost.                                     |
| `gsp_question_len`       | Integer  | `[128]`   | Length of each question appended to the system prompt, in tokens (fixed). Represents the unique, non-cacheable portion of each request.                                          |
| `gsp_output_len`         | Integer  | `[256]`   | Target length of generated outputs in tokens (fixed). Controls decoding phase duration and KV cache growth during generation.                                          |

---
In the evaluation of original paper, authors synthesize workloads based on two distinct datasets to capture a variety of real-world scenarios:

### Workload Datasets
- sharegpt (default): loads ShareGPT-style pairs; optionally restrict with --sharegpt-context-len and override outputs with --sharegpt-output-len

- generated-shared-prefix: synthetic dataset with shared long system prompts and short questions

<!-- ### ShareGPT Dataset

- **Characteristics:**
  - The ShareGPT dataset is a collection of user-shared conversations with ChatGPT.
  - It features significantly longer input prompts (approximately 8.4× longer) and outputs (about 5.8× longer) compared to other datasets, with high variance.
- **Implications for Evaluation:**
  - The long prompts and outputs place a greater demand on memory management (e.g., KV cache handling).
  - Evaluations with ShareGPT stress-test the system’s ability to handle complex, high-variance workloads under high concurrency. -->



## Performance Metrics

When evaluating vLLM, focus on the following key performance metrics:

- **Throughput (Tokens/sec):** The number of tokens processed per second, which reflects the overall efficiency of the inference engine.
- **Time to First Token (TTFT):** The latency from when a request is received to when the first token is generated.
- **Token Processing Time (TPOT):** The average time taken to generate each subsequent token.
- **Normalized Latency:** The end-to-end latency of each request divided by the output length, which helps compare performance across requests with varying lengths.
- **Batching Efficiency:** The ratio of the actual batch size to the maximum possible batch size, reflecting scheduler effectiveness.

---

### Output Quality Metrics
- **Successful Requests Ratio:**  


- **BLEU (Bilingual Evaluation Understudy):**  
  Measures N-gram precision against reference text with a brevity penalty.    

---

---

## Cost / Runtime Considerations

When evaluating vLLM performance, we focus on three key metrics that directly reflect resource consumption and runtime efficiency:

### 1. **cache_usage_perc_stats**  
- **Definition:** Percentage of GPU memory occupied by vLLM’s KV cache or internal buffers throughout the benchmark run.  
- **Why it matters:**  
  - Indicates memory pressure; sustained high utilization (e.g., > 90%) may risk OOM or reduced batch capacity.  
  - Helps tune parameters like `block-size`, `max-num-batched-tokens`, or enable `chunked-prefill` for efficiency.  
- **Collection method:** Exposed via the `vllm:cache_usage_perc` metric (0.85 means 85% usage) :contentReference[oaicite:1]{index=1}.  

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

### 4. **prefix_cache_hits_total**  

---

### Summary Table

| Metric                         | Unit     | Significance                                                                 |
|-------------------------------|----------|------------------------------------------------------------------------------|
| `gpu_cache_usage_perc_stats`  | %        | Memory pressure and capacity for batching                                   |
| `process_cpu_seconds_stats`   | seconds  | CPU overhead from scheduling, tokenization, and memory management           |
| `benchmark_duration_s`        | seconds  | End-to-end runtime reflecting overall system efficiency and throughput      |

---
