# AutoGPT

## Configurable PARAMETERS

> Source of truth: `classic/forge/forge/agent/base.py`, `classic/forge/forge/config/`,
> `classic/original_autogpt/autogpt/app/config.py`.

### AI Components

#### Agent (Core Cognitive Module)
These parameters define the core reasoning capabilities, behavioral traits, and operational boundaries of the AutoGPT agent.

| Configuration Parameter | Type | Range / Example | Source | Description |
|---|---|---|---|---|
| fast_llm | `ModelName` (union enum) | OpenAI: GPT-3.5-turbo, GPT-4, GPT-4o; Anthropic: Claude-3; Groq: Mixtral-8x7b; Llamafile: Mistral-7B | `BaseAgentConfiguration` (forge/agent/base.py:53) | Model for efficiency-focused operations. Default: `GPT3_16k` ("gpt-3.5-turbo-16k") in BaseAgentConfiguration; `GPT3` ("gpt-3.5-turbo") in AppConfig. Env: `FAST_LLM`. |
| smart_llm | `ModelName` (union enum) | Same as fast_llm | `BaseAgentConfiguration` (forge/agent/base.py:54) | Model for complex reasoning. Default: `GPT4` ("gpt-4") in BaseAgentConfiguration; `GPT4_TURBO` ("gpt-4-turbo") in AppConfig. Env: `SMART_LLM`. |
| use_functions_api | `bool` | {True, False} | `BaseAgentConfiguration` (forge/agent/base.py:55) | Toggle OpenAI Functions/tool-calling API vs free-form JSON parsing for command invocation. Default: False. Env: `OPENAI_FUNCTIONS`. Anthropic provider auto-disables it (see `agent.py:106-110`). |
| big_brain | `bool` | {True, False} | `BaseAgentConfiguration` (forge/agent/base.py:60) | When True uses `smart_llm` for thinking; when False uses `fast_llm`. Default: True. |
| ai_profile | `AIProfile` | {ai_name: str, ai_role: str, ai_goals: list[str]} | `BaseAgentSettings` (forge/agent/base.py:132), `forge/config/ai_profile.py` | Agent identity and objectives. |
| directives | `AIDirectives` | {resources: list[str], constraints: list[str], best_practices: list[str]} | `BaseAgentSettings` (forge/agent/base.py:135), `forge/config/ai_directives.py` | Guiding principles, constraints, and best practices. Field name in code is `directives`, not `ai_directives`. |
| cycle_budget | `Optional[int]` | {0, 1, ..., None} | `BaseAgentConfiguration` (forge/agent/base.py:66) | Number of unsupervised execution cycles. `None` = unlimited (but the benchmark client caps it at 200 steps via `max_steps = cycle_budget or 200` in `mf_sampler/agent_runner.py`), `1` = step-by-step, `0` = terminate. Default: `1`. In original_autogpt, derived from `--continuous` + `--continuous-limit` CLI flags. Related: `cycles_remaining` (default 1) and `cycle_count` (default 0) track runtime state. |
| send_token_limit | `Optional[int]` | None (defaults to 75% of llm.max_tokens) | `BaseAgentConfiguration` (forge/agent/base.py:81) | Token limit for prompt construction. |

---

### Non-AI Components

#### Component Architecture
Components are modular building blocks that extend the agent's functionality. They are **auto-collected** at instantiation time — any instance attribute that is a subclass of `AgentComponent` is automatically registered (via `AgentMeta.__call__` → `_collect_components()`). Individual components can be disabled via their `_enabled` flag.

| Component | Class | Protocols Implemented | Source |
|---|---|---|---|
| File Manager | `FileManagerComponent` | DirectiveProvider, CommandProvider | forge/components/file_manager/ |
| Web Search | `WebSearchComponent` | DirectiveProvider, CommandProvider | forge/components/web/search.py |
| Web Selenium | `WebSeleniumComponent` | DirectiveProvider, CommandProvider | forge/components/web/selenium.py |
| Code Executor | `CodeExecutorComponent` | CommandProvider | forge/components/code_executor/ |
| Image Generator | `ImageGeneratorComponent` | CommandProvider | forge/components/image_gen/ |
| User Interaction | `UserInteractionComponent` | CommandProvider | forge/components/user_interaction/ |
| Git Operations | `GitOperationsComponent` | CommandProvider | forge/components/git_operations/ |
| System | `SystemComponent` | DirectiveProvider, MessageProvider, CommandProvider | forge/components/system/ |
| Context | `ContextComponent` | MessageProvider, CommandProvider | forge/components/context/ |
| Action History | `ActionHistoryComponent` | MessageProvider, AfterExecute, AfterParse | forge/components/action_history/ |
| Watchdog | `WatchdogComponent` | AfterParse | forge/components/watchdog/watchdog.py |

**Component Protocols** (code interfaces in `forge/agent/protocols.py`, NOT user-configurable):
- `DirectiveProvider` — provides constraints, resources, best practices
- `CommandProvider` — provides executable commands
- `MessageProvider` — provides chat messages for the prompt
- `AfterParse` — hook after parsing LLM response
- `ExecutionFailure` — hook on execution failure
- `AfterExecute` — hook after command execution

| Configuration Parameter | Type | Source | Description |
|---|---|---|---|
| component_config_file | `Optional[Path]` | `AppConfig` (original_autogpt/app/config.py:42) | Path to JSON file with component-specific configurations. Env: `COMPONENT_CONFIG_FILE`. |

---

### Memory Module
These parameters control how the agent stores and retrieves action history.

| Configuration Parameter | Type | Default | Source | Description |
|---|---|---|---|---|
| ActionHistory.max_tokens | `int` | 1024 | `ActionHistoryConfiguration` (forge/components/action_history/action_history.py:21) | Max tokens for history message generation. |
| ActionHistory.full_message_count | `int` | 4 | Same file, line 25 | Number of latest non-summarized messages in history. |
| ActionHistory.llm_name | `ModelName` | GPT-3.5-turbo | Same file, line 19 | LLM used to compress/summarize history. |
| ActionHistory.spacy_language_model | `str` | "en_core_web_sm" | Same file, line 23 | Spacy model for summary chunking. |

**Runtime override:** When `Agent.__init__` instantiates `ActionHistoryComponent`, it overrides the default `max_tokens` with `self.send_token_limit` (see `original_autogpt/autogpt/agents/agent.py:116-127`). This means the `send_token_limit` agent parameter effectively controls the history token budget at runtime, not only `ActionHistoryConfiguration.max_tokens`.

The agent maintains an `EpisodicActionHistory` (list of `Episode` objects) that grows unboundedly — there is no configurable max-items limit.

Pipeline execution tracing is always active via `BaseAgent._trace` (a list of strings); there is no Boolean toggle to disable it.

---

### Environment Interface
These parameters define the agent's interaction with external systems.

| Configuration Parameter | Type | Default | Source | Description |
|---|---|---|---|---|
| allow_fs_access | `bool` | False | `BaseAgentConfiguration` (forge/agent/base.py:51) | Grants or restricts file system access. |
| shell_command_control | `Literal["allowlist","denylist"]` | "allowlist" | `CodeExecutorConfiguration` | Controls which shell commands are permitted. |
| shell_allowlist / shell_denylist | `list[str]` | [] | `CodeExecutorConfiguration` | Explicit allow/deny lists for shell commands. |

Note: there is no standalone "toolset" parameter. Available tools (commands) are determined by which `CommandProvider` components are enabled and their `get_commands()` output.

---

## Two-Level Sampling Design

Benchmarking uses a two-level loop structure. Every Agent configuration (outer loop) is tested against the **same** set of workload fidelity points (inner loop), enabling fair cross-agent comparison.

**Current design target (single-model, Plan B runtime-optimized).** Outer 2,304 (full grid: 16 AI × 144 Non-AI) × Inner {28 or 112} total points, selectable at runtime via `--inner-loop {plan_b,full}`. LHS down-samples the outer dimension for quick / medium runs.

| `--inner-loop` | Inner pts | `--quick` (16 LHS) | `--medium` (64 LHS) | `--full` (2,304 grid) |
|---|---:|---:|---:|---:|
| `plan_b` (default) | **28** | 448 | 1,792 | **64,512** |
| `full` | **112** | 1,792 | 7,168 | **258,048** |

Source: `classic/benchmark/mf_sampler/examples/large_scale_example.py`.

> **Plan B (default)** compresses the inner loop from 112→28 (drop `MODERATE`/`MULTI_STAGE`, keep `requests_count ∈ {1,3}`), lowers `cycle_budget` to `[3, 10, 25]`, and adds a 180s **task wall-clock timeout** (`ExecutionConstraints.task_max_duration_s`, enforced client-side in `mf_sampler/agent_runner.py`). With single-task avg duration ~38s on a 7B model, this drops the full outer-grid run from ~2.85 yr → ~57 d, and `--quick` LHS from 7 d → ~9.5 h.
>
> **`--inner-loop full`** restores the original `4 task_types × 4 requests_counts × 7 categories = 112` cartesian (e.g. for an SNR / coverage baseline against Plan B). It uses the **same `results/large_scale/` output dir** as Plan B because (i) `workload_id` and `agent_config_id` are deterministic functions of their parameters (`sampler.py::_workload_id_from`, `_agent_config_id`), (ii) LHS uses a fixed seed = 42 (`sampler.py:181`), and (iii) Plan B's 28 workloads are a strict subset of full's 112. Re-running with `--inner-loop full` therefore **resumes** the existing Plan B samples and only spends new compute on the 84 added workload IDs (`MODERATE_*`, `MULTI_STAGE_*`, `*_r2_*`, `*_r4_*`). Pass `--no-resume` for a clean re-run.

### Outer Loop — Agent Configuration Space

Parameters that define the AI system under test. Divided into **AI parameters** (core cognition/LLM) and **Non-AI parameters** (tools/environment/memory).

#### AI Component Parameters (16 combinations in single-model mode)

Single-model deployment grid: `1 model × 1 big_brain × 2 send_token_limits × 1 cognitive_strategy × 2 temperatures × 2 max_output_tokens × 1 prompt_style × 2 use_functions_apis = 16`. With dual-model (`big_brains=[True, False]`) it becomes 32.

| Factor | Source-code Parameter | Type | Range / Notes |
|---|---|---|---|
| **LLM Model** | `fast_llm`, `smart_llm` | `ModelName` | See supported model enums above. Currently fixed to `qwen2.5-7b-instruct` (single-model vLLM) |
| **Big Brain** | `big_brain` | `bool` | True → smart_llm for thinking; False → fast_llm. **Fixed = True** in single-model mode (toggling has no effect when `smart_llm == fast_llm`) |
| **Prompt Token Limit** | `send_token_limit` | `Optional[int]` | [None, 2048] — `None` defaults to 75% of model max_tokens |
| **Temperature** | `BaseAgentConfiguration.llm_temperature` | `float` | [0.0, 0.7] — deterministic baseline vs. creative regime. The previously-swept levels [0.3, 1.0] were dropped (low signal-to-noise mid-range / extreme values produce degenerate outputs); the saved sample budget is reinvested into denser Non-AI coverage. Passed to LLM API `temperature` param. |
| **Max Output Tokens** | `BaseAgentConfiguration.llm_max_output_tokens` | `int` | [512, 2048] — replaces previous [256, 1024]: 256 was systematically too small for CODE_GENERATION (non-trivial functions routinely need 400+ tokens), and 2048 leaves headroom for MULTI_STAGE summaries. Passed to LLM API `max_tokens` param. |
| **Prompt Style** | *(experiment-design)* | Enum | **Fixed: CONCISE** (only affects DIRECT_LLM mode; in Agent Protocol mode the prompt is built by `OneShotAgentPromptStrategy`). |
| **Use Functions API** | `use_functions_api` | `bool` | [True, False] — True routes commands via the OpenAI Functions/tool-calling API; False uses free-form JSON parsing. Materially affects both token consumption and command-parse failure rate. Source: `forge/agent/base.py:55`. Requires vLLM to be started with `--enable-auto-tool-choice --tool-call-parser hermes` (see "Local Single-Model Deployment" below). |

Note: `cognitive_strategy` is **fixed to ONE_SHOT** in Agent Protocol mode (AutoGPT's `OneShotAgentPromptStrategy` is hardcoded at `original_autogpt/autogpt/agents/agent.py:103,111`).

Note: `cycle_budget` was previously listed here but has been moved to **Non-AI** because it controls outer agent-loop orchestration (number of step iterations) rather than per-call LLM cognition. See Non-AI table below.

#### Non-AI Component Parameters (144 combinations)

Single-model deployment grid: `4 component_sets × 2 allow_fs × 3 full_msg_count × 2 shell_cmd_ctrl × 3 cycle_budgets = 144`.

| Factor | Source-code Parameter | Type | Range / Notes |
|---|---|---|---|
| **Enabled Components** | Component `_enabled` flags | Subset of components | 4 tiers: FileManager only → +CodeExecutor → +WebSearch → +Context |
| **File System Access** | `allow_fs_access` | `bool` | [True, False] — controls whether the agent can directly read/write the file system |
| **Full Message Count** | `ActionHistoryConfiguration.full_message_count` | `int` | [2, 4, 8] — number of latest non-summarized messages in history (short-term memory depth) |
| **Shell Command Control** | `CodeExecutorConfiguration.shell_command_control` | `str` | ["allowlist", "denylist"] — permission mode for shell command execution. The concrete command lists live in the "Shell command lists" section below. `execute_local_commands=True` is forced on as a constant in `MultiFidelityConfig`; with the default (False), shell commands never enter the agent command table at all, which would silently make this dimension a no-op. |
| **Cycle Budget** | `cycle_budget` | `Optional[int]` | [3, 10, 25] — upper bound on agent main-loop iterations (an execution-orchestration parameter; it does not affect per-call LLM cognition). Plan B lowers the cap from 100 to 25 to bound worst-case per-task wall time, in concert with `task_max_duration_s=180s`. The client also has a fallback `max_steps = cycle_budget or 200` in `mf_sampler/agent_runner.py`. |
| **Task Wall Timeout** | `ExecutionConstraints.task_max_duration_s` | `Optional[float]` | **Constant 180s (not part of the sweep)** — client-side wall-clock cap; on expiry returns `status="timeout"`. AutoGPT's self-termination rate is extremely low (`cycle_efficiency ≈ 0.5%`), so without this cap a high-`cycle_budget` task can monopolise the GPU for tens of minutes. Defined in `mf_sampler/config.py::ExecutionConstraints`; enforced in `mf_sampler/agent_runner.py` — before each step the loop checks elapsed time and clamps `asyncio.wait_for(timeout=...)` to `min(step_timeout, remaining_wall)`. |

**Note on `full_message_count` ↔ `send_token_limit` coupling:** `full_message_count` indirectly modulates LLM cognition through prompt content. Together with the AI-side `send_token_limit` it **jointly determines the total prompt size** — e.g. when `send_token_limit=2048` and `full_message_count=8` there may not be room for 8 complete messages, in which case `full_message_count` degrades into a soft upper bound. The two parameters should not be interpreted independently.

**Implicit dependency `enabled_components × allow_fs_access`:** Source `forge/agent/base.py:348` **force-disables `FileManagerComponent`** whenever `allow_fs_access=False`, regardless of the `enabled_components` setting. Consequently, when `allow_fs_access=False`, all 4 component tiers collapse into "no FM + the remaining 0–3 components" — about half of the 144 combinations are semantically compressed. Experimental analysis should therefore not attribute "FM on/off" effects to the `enabled_components` dimension.

**Shell command lists (not part of the sweep — applied as constants to every sample point):**

`forge/components/code_executor/code_executor.py:224` `validate_command()` only matches `shlex.split(command_line)[0]` (the command name) — it does not constrain arguments. Because the benchmark runs entirely inside the Docker `{agent_id}_sandbox`, the practical purpose of these two lists is **to test whether the agent obeys the stated policy**, not to provide physical isolation.

| Mode | Command list (21 / 23 entries) |
|---|---|
| `shell_allowlist` | `ls`, `cat`, `head`, `tail`, `grep`, `find`, `wc`, `sort`, `uniq`, `echo`, `pwd`, `mkdir`, `touch`, `cp`, `mv`, `python`, `python3`, `pip`, `pip3`, `make`, `git` — **`rm` is intentionally excluded** to force the agent to use `file_manager`. |
| `shell_denylist` | `sudo`, `su`, `shutdown`, `reboot`, `poweroff`, `halt`, `dd`, `mkfs`, `fdisk`, `parted`, `curl`, `wget`, `nc`, `ncat`, `telnet`, `ssh`, `scp`, `rsync`, `chmod`, `chown`, `kill`, `killall`, `pkill` — allows `rm` / `python` / `make` and most other day-to-day commands. |

Example observable difference: `rm tmpfile` is rejected in `allowlist` mode but accepted in `denylist` mode; `curl https://...` is rejected in both (denylist bans it explicitly, allowlist simply omits it). The constants are centrally defined in `mf_sampler/config.py` as `MultiFidelityConfig.shell_allowlist / shell_denylist`, propagated through `agent_config_overrides` to `BaseAgentConfiguration.shell_allowlist/denylist` (`forge/agent/base.py`), and finally injected into `CodeExecutorConfiguration`.

Note: `ActionHistoryConfiguration.max_tokens` (default 1024) is **fully overridden at runtime** by `send_token_limit` — see `original_autogpt/autogpt/agents/agent.py:118` (`history_config_kwargs["max_tokens"] = self.send_token_limit`) — so it is not exposed as a tunable factor.

#### Local Single-Model Deployment (Current Setup)

When GPU VRAM is insufficient for two models (e.g., 7B + 3B concurrently), deploy a single model for both `smart_llm` and `fast_llm`:

| Setting | Value |
|---|---|
| `SMART_LLM` | `qwen2.5-7b-instruct` |
| `FAST_LLM` | `qwen2.5-7b-instruct` (same as smart) |
| `big_brains` | `[True]` (fixed — toggling has no effect with single model) |

Run one vLLM instance:
```bash
vllm serve /path/to/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8080 \
  --served-model-name qwen2.5-7b-instruct \
  --gpu-memory-utilization 0.85 \
  --dtype auto --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes
```

> **`--enable-auto-tool-choice` + `--tool-call-parser hermes` are mandatory** (introduced in vLLM ≥ 0.6; verified working on 0.16.0). When `use_functions_api=True`, AutoGPT (`one_shot.py:272-275`) **hard-asserts** that `response.tool_calls` is non-empty; otherwise it raises `InvalidAgentResponseError("Assistant did not use a tool")`. Qwen2.5-Instruct was trained to emit tool calls in the `<tool_call>{...}</tool_call>` format (same family as Hermes2Pro), so `hermes` is the correct parser.
>
> Without these two flags, the 8 `use_functions_api=True` agent configs × 144 Non-AI × 28 inner-loop points = **32,256 sample points fail 100%** — wasting half of any LHS / full-grid budget. Use `classic/benchmark/scripts/start_vllm.sh` to launch with the right flags, and `classic/benchmark/scripts/verify_tool_calls.sh` to confirm that `tool_calls` is being populated correctly.

If dual-model deployment becomes feasible (e.g., with more VRAM), set `FAST_LLM=qwen2.5-3b-instruct`, enable `OPENAI_MODEL_BASE_URLS`, and change `big_brains=[True, False]` in the sampling config.

#### Server-side Log Capture (vLLM + AutoGPT)

The benchmark itself only captures the `mf_sampler.*` logger output and a hardware-monitor time-series; per-step vLLM messages and AutoGPT Agent Protocol traces would otherwise be lost in the side-car processes' stdout. A three-piece tee-and-slice pipeline closes this gap:

| Piece | File | Behaviour |
|---|---|---|
| **vLLM launcher** | `classic/benchmark/scripts/start_vllm.sh` | tee's stdout+stderr to `classic/benchmark/logs/vllm.log` and exports `BENCHMARK_VLLM_LOG_PATH` |
| **AutoGPT launcher** | `classic/benchmark/scripts/start_autogpt_server.sh` | tee's stdout+stderr to `classic/benchmark/logs/autogpt_server.log` and exports `BENCHMARK_AUTOGPT_LOG_PATH` |
| **Sampler bookkeeping** | `mf_sampler/sampler.py` `_server_log_snapshot` | Records `[start_byte, end_byte, start_ts, end_ts]` for each tracked log per sample; payload key `server_log_offsets` in every `fidelities/{wl_id}/{agent_config_id}.json` |
| **Slicer CLI** | `classic/benchmark/scripts/extract_sample_logs.py` | `extract_sample_logs.py path/to/sample.json [--source vllm|autogpt|all] [--write] [--tail N]` reconstructs the exact server-side window for one sample |

Important: the sampler must be launched from a shell that has the two `BENCHMARK_*_LOG_PATH` env vars set (the `start_*.sh` scripts only export them within their own shell — for the sampler shell, `source`-able env or manual `export` is needed). When the env vars are unset the system silently degrades to pre-change behaviour (no `server_log_offsets` field in the JSON, slicer prints a NOTE).

The sample JSON also now persists `task_results` (per-task `output`, `error`, `metadata.timeout_type`, `metadata.num_steps`, `token_usage`, ...). Combined with the sliced server logs, a wall-timeout or `InvalidAgentResponseError` sample can be root-caused without re-running.

### Inner Loop — Workload Fidelity Space (workload-intrinsic properties)

These factors describe the **workload itself** — task complexity, scale, and domain. They are independent of the AI system configuration.

In the normalized LLMSYS-HPOBench data, these workload factors are encoded in the fidelity directory and CSV file name. They are not duplicated as `FIDELITY_*` columns.

The filename order follows `experiment-data/tab-format.tex`:

```text
{task_type}-req{requests_count}-{workload_category}
```

Example:

```text
moderate-req1-code_generation/moderate-req1-code_generation.csv
```

Each row also links its hardware snapshot and run log through `hw-file` and `log-file`, with artifacts stored as:

```text
hw_file/hw-{ID}.txt
log_file/log-{ID}.txt
```

| Factor | Source | `--inner-loop plan_b` (default) | `--inner-loop full` |
|---|---|---|---|
| **Task Type (Difficulty)** | `mf_sampler/config.py::TaskType` (4-level enum: simple / moderate / complex / multi_stage). Conceptually similar to agbenchmark's `DifficultyLevel` (`agbenchmark/utils/data_types.py:7-14`) but **with no direct source-code mapping**. | **[SIMPLE, COMPLEX]** — only the two extremes; MODERATE / MULTI_STAGE have low SNR vs their neighbours and were dropped to halve the inner loop. | **[SIMPLE, MODERATE, COMPLEX, MULTI_STAGE]** — full enum span (4 values). |
| **Requests Count (Workload Size)** | `FidelityConfig.requests_count` (`mf_sampler/config.py:365`, free positive int). | **[1, 3]** — minimal-batch (1) and medium-batch (3); 2 / 4 were dropped because their marginal effect is below measurement noise. | **[1, 2, 3, 4]** — original Plan A sweep. |
| **Workload Category** | `mf_sampler/config.py::WorkloadCategory` (7-value enum). | **All 7** retained: math_reasoning, code_generation, logic_puzzles, data_analysis, memory_retrieval, instruction_adherence, text_classification. Each is an orthogonal domain — dropping any creates a coverage gap, not noise reduction. | **All 7** (identical to Plan B). |
| **Cartesian product** | — | **2 × 2 × 7 = 28** workload fidelity points. | **4 × 4 × 7 = 112** workload fidelity points. |

The inner loop always uses the full cartesian product **of whichever set is selected** to ensure complete coverage on those axes; LHS / sampling-strategy options apply only to the outer agent-config loop.

Key considerations for Requests Count:
- Low request counts (r=1) focus on correctness, instruction adherence and per-request quality.
- Higher request counts (r=3, and r=2 / r=4 in `--inner-loop full`) evaluate steady-state behavior, memory/context handling, and average latency.

#### What `InformationAvailability` is *not*

`mf_sampler/config.py::InformationAvailability` (3-value enum: `INTERNAL_ONLY` / `EXTERNAL_WEB` / `MIXED`) is defined in source and serialised in `FidelityConfig`, but **is not part of the inner-loop sweep** in either mode (its declaration is annotated *"Optional metadata (unused — kept for serialization compatibility)"*). Counting it would inflate the theoretical maximum to `4 × 4 × 7 × 3 = 336`, but no `MultiFidelityConfig` field drives it, so the practical full cartesian remains 112.

<!-- ### Design rationale

- **Temperature, max_tokens, prompt_style** are properties of the AI system (how the LLM generates output and how prompts are styled), not of the workload. Changing temperature does not change *what task* is being solved, only *how* the system approaches it.
- The outer loop may use Latin Hypercube Sampling (LHS) when the full factorial grid is too large; the inner loop always uses full cartesian product to ensure every workload combination is tested for each agent.
- Report both aggregate metrics (total completion rate, total time) and per-request distributions (latency percentiles, per-request token usage). -->

---

## Workloads

The benchmark uses **7 precisely-evaluable workload categories** defined in `mf_sampler/config.py:88-102` (`WorkloadCategory` enum). They are inspired by `agbenchmark` challenge categories (`Category` enum in `agbenchmark/utils/data_types.py:31-39`: general, data, coding, scrape_synthesize, web, GAIA_1, GAIA_2, GAIA_3) but are tailored for programmatic scoring (open-ended categories like text summarization / creative writing / translation / safety-ethics are excluded because they require LLM-as-judge).

| # | `WorkloadCategory` | Task focus | Tools relied on |
|---|---|---|---|
| 1 | `MEMORY_RETRIEVAL` | Extract / filter / retain specific information (e.g., IDs) from files or web sources, with or without distracting noise. | File manager, web scraper, action history |
| 2 | `CODE_GENERATION` | Write functional code (e.g., "two-sum") or deploy small apps (e.g., Flask health-check server). | Code executor, file editor, shell |
| 3 | `INSTRUCTION_ADHERENCE` | Strictly follow constrained directives (e.g., create exactly 5 files from an array — no extras). | File manager, action validation |
| 4 | `MATH_REASONING` | Solve symbolic / arithmetic / step-by-step math problems with verifiable answers. | LLM only (optionally code executor for arithmetic) |
| 5 | `LOGIC_PUZZLES` | Solve constraint-satisfaction puzzles, deduction tasks, structured logical reasoning. | LLM only |
| 6 | `DATA_ANALYSIS` | Read structured data (CSV/JSON), aggregate / filter / compute summary statistics. | File manager, code executor |
| 7 | `TEXT_CLASSIFICATION` | Classify text into a fixed label set (e.g., emotion / topic / toxicity buckets). | LLM only |

All 7 categories are exercised by the outer-loop sampling configuration (`large_scale_example.py`, `workload_categories=[...]`).

---

## Measurement Dimensions

Benchmarking metrics are divided into three orthogonal dimensions: **Performance** (quality), **Cost** (resource consumption), and **Throughput** (efficiency). This separation ensures that "how well did the agent do" is never conflated with "how much did it cost".

### Performance — Task Quality & Reliability

Measures *how well* the agent completes tasks, independent of resource usage.

| Metric | Type | Description |
|---|---|---|
| **Success Rate** | float (0–1) | Fraction of tasks completed successfully |
| **Error Rate** | float (0–1) | Fraction of tasks that resulted in errors |
| **Timeout Rate** | float (0–1) | Aggregate timeout fraction = `step_timeout_rate + wall_timeout_rate`. Kept for back-compat / overview |
| **Step Timeout Rate** | float (0–1) | Fraction of tasks where a single step (one `POST /steps` = 1 LLM inference + 1 tool invocation) exceeded `timeout_per_step` (default 300s). **Indicates**: vLLM inference stall, oversize single generation (`max_tokens=2048`), tool / Docker subprocess hang, or network jitter. Source: `mf_sampler/agent_runner.py` (`metadata.timeout_type == "step"`). |
| **Wall Timeout Rate** | float (0–1) | Fraction of tasks where cumulative wall-clock time per task exceeded `task_max_duration_s` (default 180s, Plan B). **Indicates**: agent retry storms, infinite loops, `cycle_efficiency ≈ 0.5%`, or `cycle_budget` mismatched to task difficulty. Source: `metadata.timeout_type == "wall"`. |
| **Avg Steps per Wall Timeout** | Optional[float] | Mean number of steps consumed before a wall timeout fired; `None` when there were no wall timeouts in this sample. **Diagnostic**: a value close to `cycle_budget` means the agent ran out the budget without converging — lower `cycle_budget` or improve the prompt; a small value (3–4) means individual steps were too slow — lower `max_output_tokens` or `temperature`. |
| **Instruction Adherence** | float (0–1) | How closely the agent follows constraints and directives |
| **Num Tasks / Success / Failures** | int | Absolute counts for the workload |

### Cost — Resource Consumption

Measures *what resources* the agent consumed. For local models, monetary cost (`cost_usd`) is 0; use token-based and time-based proxies instead.

| Metric | Type | Unit | Description |
|---|---|---|---|
| **total_tokens** | int | tokens | Total tokens consumed (prompt + completion). Primary cost proxy for local models. |
| **prompt_tokens** | int | tokens | Input/prompt tokens consumed |
| **completion_tokens** | int | tokens | Output/completion tokens consumed |
| **avg_tokens_per_task** | float | tokens/task | Average token cost per task |
| **cost_usd** | float | USD | Estimated monetary cost (0 for local models) |
| **avg_duration_s** | float | seconds | Average wall-clock time per task. Primary time-cost proxy. |
| **total_duration_s** | float | seconds | Total wall-clock time for all tasks |
| **latency_p50 / p95 / p99** | float | seconds | Per-task latency percentiles |

Cost drivers in AutoGPT source code:
- **LLM Cost:** Driven by `smart_llm` vs. `fast_llm` usage; `big_brain` flag toggles preference (forge/agent/base.py:60).
- **Cycle Overhead:** More cycles (`cycle_budget`) → more LLM calls → more tokens & time.
- **Tool Execution Time:** External tools (web scraping, code execution) add latency beyond LLM inference.
- **Memory Management Cost:** Action history increases prompt size; the effective budget is `send_token_limit` (which overrides `ActionHistoryConfiguration.max_tokens` at runtime — see `agent.py:118`), modulated by `full_message_count`.
- **Component Retry Cost:** Pipeline retry (`run_pipeline(retry_limit=3)`, base.py:196) and LLM API retry (`retries_per_request=7`, schema.py:194) add redundant token usage on failure.

### Throughput — Efficiency

Derived metrics combining output and cost: how much useful work per unit of resource.

| Metric | Unit | Description |
|---|---|---|
| **throughput_tasks_per_sec** | tasks/s | Task completion rate |
| **throughput_tokens_per_sec** | tokens/s | Token processing rate |

### Experiment design recommendation

- Report all three dimensions for each (agent_config, workload_fidelity) pair.
- Sweep Requests Count as a fidelity axis (Plan B uses [1, 3]; `--inner-loop full` extends to [1, 2, 3, 4]) and produce plots showing how performance, cost, and throughput change with workload size.
- For Pareto analysis, plot performance (y) vs. cost (x) to find optimal config-cost tradeoffs.
- To validate that Plan B compression did not erase signal, run `--quick --inner-loop full` (1,792 pts, ~19 h) once and compare the per-agent_config metric distributions on the 28 shared workloads against the 84 added ones — if the marginal contribution of MODERATE/MULTI_STAGE/r∈{2,4} to your target metric is below noise, the Plan B grid is safe for the full sweep.
