# PolyServe

**PolyServe helps you run LLMs on your own hardware without tuning an inference backend by hand.** Give it a model, and it checks your machine, benchmarks the available options, and serves the chosen configuration through an OpenAI-compatible API.

```bash
pip install git+https://github.com/Aagam-Bothara/polyserve.git   # not yet on PyPI
polyserve serve meta-llama/Llama-3.2-3B-Instruct            # --objective balanced
curl localhost:8000/v1/chat/completions -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

First launch: discover hardware → prepare model → calibrate once → serve.
Later launches: load the cached profile → serve.

PolyServe works with vLLM, SGLang and llama.cpp. It handles the setup questions that usually take trial and error: which backend to use, which quantization fits, and how to set memory and batch sizes for your workload. It makes those choices by running benchmarks on your machine.

---

## Supported hardware (v1)

| Hardware | Backends tried | Status |
|---|---|---|
| NVIDIA, compute capability ≥ 7.5 (Turing and newer) | vLLM, SGLang, llama.cpp (CUDA) | vLLM and llama.cpp **benchmarked** on an RTX 3090; SGLang implemented, **never benchmarked** |
| NVIDIA, compute capability < 7.5 (Pascal, Volta) | llama.cpp (CUDA) | implemented, **never benchmarked** |
| x86 CPU | llama.cpp; vLLM-CPU if AVX-512 is present | implemented, **never benchmarked** |

PolyServe runs on Linux with Python 3.10–3.13. The table above lists the backend candidates for each type of hardware; the benchmarks below show what has been measured so far.

Install the backends you want to try. PolyServe starts each one as a separate process and supplies the settings it has tuned:

```bash
pip install "polyserve[nvml] @ git+https://github.com/Aagam-Bothara/polyserve.git"   # + NVML telemetry
pip install "vllm==0.11.0" "transformers>=4.56,<5"   # vLLM 0.11 breaks on transformers 5.x
pip install sglang                   # optional
export LLAMA_SERVER=/path/to/llama-server   # build llama.cpp with -DGGML_CUDA=ON; optional if on PATH
```

---

## How it works

```mermaid
flowchart LR
    P[probe<br/>GPU / CPU / installed backends] --> S[select<br/>candidate backends]
    S --> M[prepare model<br/>HF weights or GGUF]
    M --> PL[memory planner<br/>prunes the grid to what fits]
    PL --> C[calibrate<br/>staged search, llmtrace-measured]
    PR[performance predictor<br/>roofline, fitted per machine] -.prunes.-> C
    C --> CA[(profile cache<br/>~/.polyserve/profiles)]
    CA --> SV[serve<br/>supervised backend + OpenAI proxy :8000]
```

1. **Check your hardware.** PolyServe detects your GPU, its compute capability and VRAM, CPU cores and AVX2/AVX-512 support, RAM, and which backends can run. Use `polyserve probe` to inspect the resulting `HardwareDescriptor`.

2. **Choose candidate backends.** The hardware rules above determine which backends to try. If several are available, calibration decides which one to use.

3. **Prepare the model.** vLLM and SGLang use Hugging Face weights directly. For llama.cpp, PolyServe finds a pre-quantised GGUF on the Hub (Q4_K_M, Q5_K_M, Q6_K or Q8_0), or converts and quantises FP16 weights locally. It only prepares the quantizations that pass memory planning.

4. **Check what fits in memory.** Before launching a backend, the planner estimates its memory needs:

   `estimated = weights + kv_cache(ctx, batch, dtype) + runtime_workspace + safety_margin`

   It keeps a configuration only if `estimated ≤ 0.95 × available`. On a 24 GB card, this reduced 54 vLLM candidates to 48 for a 3B model at 4k context, and to 18 at 32k. None of the admitted configurations ran out of memory in that run. Use `polyserve plan <model>` to see what fits before launching anything.

   Each trial records the estimate alongside measured peak memory from NVML and the backend's own weight, KV cache and workspace figures. `polyserve memory-report` shows the errors by backend. Add `--apply` to replace the initial workspace constant and 5% margin with values fitted to your machine.

5. **Benchmark the candidates.** Calibration starts with a short run per quantization and keeps the top two. It then selects the largest safe memory configuration and tries different batch sizes and concurrency levels.

   A performance predictor helps narrow the search. Its roofline model accounts for weight and KV bandwidth, decode and prefill compute, and queueing beyond the available slots. `polyserve fit` fits three efficiency parameters per backend using your machine's trials. The predictor skips quantizations expected to perform far below the best and tests promising batch settings first. `polyserve predict` shows the predictions without launching a backend; `fit` reports leave-one-out error and rank correlation so you can check their accuracy.

   Each trial replays a fixed synthetic workload. The default uses 16 prompts, a 256-token prefill and a 128-token decode at concurrency 1/4/8, taking roughly 10 seconds. [llmtrace](https://github.com/Aagam-Bothara/llmtrace) measures tokens per second, time to first token (TTFT), time per output token (TPOT), peak memory, GPU utilisation and power.

6. **Save the results.** The chosen configuration, full launch arguments, calibration table and versions are saved in `~/.polyserve/profiles/<hardware_hash>/<model>/<objective>[-<workload>].json`. A hardware or backend version change invalidates the profile. You can also force a fresh run with `polyserve recalibrate`.

7. **Start serving.** PolyServe runs the chosen backend as a supervised process, with health checks and automatic restarts. A proxy on `:8000` exposes `/v1/chat/completions`, `/v1/completions` and `/v1/models`, with streaming passthrough. Visit `/polyserve/profile` to see the active configuration and calibration table.

### Workloads

An interactive chat and a long document query need different settings. Use `--workload` to choose the synthetic traffic used during calibration. Each preset has a time-to-first-token (TTFT) limit for the `balanced` objective. PolyServe caches profiles separately for each workload, so `serve --workload rag` and `serve --workload chat` each get their own calibration.

| `--workload` | prefill | decode | concurrency | TTFT ceiling | shaped like |
|---|---|---|---|---|---|
| `default` | 256 | 128 | 1 / 4 / 8 | 500 ms | the spec's calibration workload |
| `chat` | 512 | 128 | 1 / 4 / 8 | 500 ms | assistant turns |
| `long-context` | 8192 | 256 | 1 / 2 / 4 | 2000 ms | document Q&A, summarisation |
| `generation` | 128 | 1024 | 1 / 4 / 8 | 500 ms | code / story generation |
| `high-concurrency` | 256 | 64 | 32 / 64 / 128 | 1000 ms | many short requests |
| `rag` | 6144 | 64 | 1 / 4 / 8 | 1500 ms | retrieval-augmented answers |

Prompt lengths are exact when the model's tokenizer is available (prompts are fitted to the target token count), and output tokens are counted from the server's `usage` or the tokenizer, never from stream chunks. The planner drops any config whose context cannot hold prefill + decode.

```bash
polyserve serve meta-llama/Llama-3.2-3B-Instruct --workload chat
polyserve serve meta-llama/Llama-3.2-3B-Instruct --workload rag
polyserve serve meta-llama/Llama-3.2-3B-Instruct --workload long-context
```

### Objectives

Choose what you want to optimise. PolyServe ranks configurations using the rules below, rather than combining the measurements into a weighted score.

| `--objective` | Rule |
|---|---|
| `throughput` | Highest tokens per second |
| `latency` | Lowest TTFT while meeting the minimum tokens per second |
| `balanced` (default) | Highest tokens per second within the workload's TTFT limit; override it with `--ttft-ceiling` |

Every objective also respects a per-token latency (TPOT) ceiling, the decode-phase counterpart of the TTFT limit. Each workload carries one (50 ms for `chat` and `generation`, 150 ms for `high-concurrency`, 100 ms otherwise) and `--tpot-ceiling` overrides it, so no configuration can win by starving either phase.
| `efficiency` | Lowest joules per token while meeting the minimum tokens per second |

The minimum throughput defaults to 50% of the best observed tokens per second. Set `--tok-s-floor` to use an absolute value instead. If no configuration meets the constraint, PolyServe chooses the one that comes closest and records that in the profile.

### Energy tuning: power cap and clock lock

LLM decode is memory-bandwidth bound. Past a certain core clock the SMs wait on memory, so extra frequency buys almost no throughput while power keeps rising. `--power` adds a fourth calibration stage that looks for that point on the winning configuration:

| `--power` | What it sweeps |
|---|---|
| `off` (default) | nothing |
| `cap` | board power limit at 85%, 70% and 55% of the default |
| `clock` | locked SM clock at 85%, 70% and 55% of the maximum, snapped to supported steps |
| `both` | both sets: six points, not the cross product |

Both knobs are applied through NVML to the already-running server, so the sweep needs a single launch and every point shares the same warm engine. The objective then decides. `efficiency` takes the lowest joules per token above its throughput floor. `balanced`, `latency` and `throughput` only accept a setting whose throughput is within 2% of the uncapped result, so the energy saving costs no measurable speed unless you ask for that trade. The chosen setting gets its own cached profile, is applied when `polyserve serve` starts, and appears at `/polyserve/profile`. Every trial also records the mean SM clock, which confirms that a lock actually took effect.

Changing power or clocks needs root and affects the whole machine. PolyServe writes the original state to `~/.polyserve/power-restore.json` before the first change, restores it on exit and on SIGTERM, and `polyserve power reset` undoes it after a hard kill. `polyserve power status` shows the card's limits and whether control is permitted; when it is not, calibration records the reason and skips the stage instead of failing. **This stage is implemented and tested against a simulated NVML, and has not yet run on real hardware.**

### Prefill and decode

Prefill, which processes the prompt, is compute bound. Decode, which generates tokens, is memory-bandwidth bound. They want different settings, so PolyServe tunes them separately, and `--phases` chooses how:

| `--phases` | What happens |
|---|---|
| `unified` (default) | One engine. The batch-size sweep tunes decode concurrency; a prefill stage then sweeps the prefill knob on the winner: vLLM's chunked-prefill budget (`--max-num-batched-tokens` 2048, 8192, 16384), SGLang's `--chunked-prefill-size`, or llama.cpp's micro-batch (`-ub` 256, 1024, 2048). |
| `disaggregated` | Two vLLM engines on two GPUs, joined by KV-cache transfer (NixlConnector by default; `--kv-connector`). The prefill engine gets a large prefill budget and few sequences, the decode engine many sequences and a small budget, and with `--power` only the decode GPU is capped. A router sends each request to the prefill engine for one token, takes the KV handle it returns, and streams the decode engine's output to the client. |
| `auto` | Calibrates unified, measures the disaggregated pairs, and keeps whichever wins under the objective. On a machine that cannot disaggregate it serves unified and records why. |

Disaggregation needs vLLM, two NVIDIA GPUs and the connector's package (`pip install nixl`). With `--phases disaggregated`, PolyServe checks all three before spending a calibration and refuses with the reason. Each engine's settings, the GPU split and the connector are cached in the profile, shown at `/polyserve/profile`, and restarted together if either engine dies. **The disaggregated mode is tested end to end against fake engines that follow vLLM's KV-transfer handshake, and has not yet run on real GPUs.**

---

## CLI

```
polyserve serve <model> [--objective X] [--workload W] [--phases MODE] [--power MODE] [--tpot-ceiling MS] [--port N] [--backend NAME] [--skip-calibration]
polyserve probe                 # print HardwareDescriptor
polyserve workloads             # list workload presets
polyserve plan <model>          # print feasible configs without running them
polyserve bench <model>         # run calibration and print the table, don't serve
polyserve recalibrate <model>   # force a rerun and overwrite the cached profile
polyserve profiles              # list cached profiles
polyserve compare <model>       # PolyServe's pick vs stock defaults vs Ollama, one workload -> results JSON
polyserve report                # aggregate results: median gain over the best SLO-meeting default + plot
polyserve memory-report [--apply]  # planner prediction vs measured peak memory; --apply fits workspace + margin
polyserve predict <model>       # predicted tok/s / TTFT for every feasible config, no launches
polyserve fit [--apply]         # fit the predictor from cached calibrations; leave-one-out accuracy
polyserve power status          # GPU power limit range, supported clocks, whether control is permitted
polyserve power reset           # undo a power cap or clock lock left behind by a crashed run
```

To start serving without waiting for benchmarks, use `--skip-calibration`. This launches the first candidate backend with default settings and does not cache a profile.

---

## Benchmarks

All rows measured on one RTX 3090 (24 GB, cc 8.6) with vLLM 0.11.0, llama.cpp CUDA and Ollama 0.34 installed, driven by `polyserve compare`: PolyServe's calibrated pick and every stock default run against the same workload, minutes apart, on the same card. Telemetry via llmtrace (NVML at 100 ms). Raw results are in [benchmarks/results/](benchmarks/results/); `polyserve report` regenerates [benchmarks/RESULTS.md](benchmarks/RESULTS.md) and the throughput-vs-TTFT plot.

### Qwen2.5-3B-Instruct, `--objective balanced`

> On **one RTX 3090 with one model (Qwen2.5-3B-Instruct) across six workloads**, PolyServe improves throughput by a median of **+51%** (range +7% to +67%) over the best stock/default configuration that satisfies the requested latency SLO, winning 6 of 6.

**Read this before the table: on the GPU, this number is quantisation, not tuning.** PolyServe picked vLLM with fp8 weights in all six rows, and the baseline is vLLM's 16-bit default. A separate experiment ([below](#what-the-search-actually-contributes)) runs stock vLLM with `--quantization fp8` and nothing else tuned, and finds it captures essentially the whole gain: the configuration search adds **−1.8% to +0.9%** on top of it. The defensible claim is that PolyServe automatically finds a precision the stock defaults leave on the table, not that its search finds a better configuration. On CPU, where there is no such precision to pick, the picture reverses and the search is worth **+224%**.

| workload | PolyServe pick | tok/s | best default meeting SLO | tok/s | gain | TTFT Δ | J/token gain | calibration |
|---|---|---|---|---|---|---|---|---|
| `chat` | vLLM fp8, ctx 8192, batch 64 | **1091** | vLLM defaults | 653 | **+67%** | −3 ms | +42% | 399 s / 10 trials |
| `generation` | vLLM fp8, ctx 8192, batch 16 | **1111** | vLLM defaults | 683 | **+63%** | −7 ms | +38% | 1937 s / 10 trials |
| `default` | vLLM fp8, ctx 8192, batch 64 | **1097** | vLLM defaults | 687 | **+60%** | −8 ms | +38% | 480 s / 10 trials |
| `long-context` | vLLM fp8, ctx 32768, batch 16 | **436** | vLLM defaults | 305 | **+43%** | −1 ms | +32% | 397 s / 8 trials |
| `rag` | vLLM fp8, ctx 16384, batch 64 | **705** | vLLM defaults | 504 | **+40%** | −23 ms | +30% | 370 s / 8 trials |
| `high-concurrency` | vLLM fp8, ctx 8192, batch 64 | **3628** | vLLM defaults | 3404 | **+7%** | −126 ms | +6% | 730 s / 10 trials |

No row trades latency for throughput, but the latency picture is uneven and the p50 column above hides it. Four of the six rows differ by 1 to 8 ms at p50, which is within run-to-run noise. Two are real: `high-concurrency` is 156 ms against 282 ms at p50 and 203 ms against 1973 ms at p95, and `rag` gains 23 ms. One p95 is worse: `long-context` at 98.1 ms against the baseline's 90.0 ms. Calibrating all six workloads cost 71 minutes on this card, paid once and cached.

**What made the difference?** Most of the gain came from choosing fp8 over vLLM's default bf16 on Ampere, and matching batch and context sizes to the workload instead of using the model's 32k maximum. The benefit was smaller at high concurrency: with 128 concurrent clients, stock vLLM already kept the card busy. Throughput improved by 7%, while time to first token fell from 276 ms to 142 ms. Under the same load, llama.cpp and Ollama missed the latency target, with first-token waits of 9.6 s and 17 s respectively, because their defaults served one request at a time.

### What the search actually contributes

Measured back to back on one RTX 3090, `benchmarks/isolate_quantisation.py`, raw data in [benchmarks/isolation/](benchmarks/isolation/):

| workload | stock bf16 | stock fp8 | PolyServe | gain from quantisation | gain from the search |
|---|---|---|---|---|---|
| `default` | 721 | 1139 | 1118 | **+58.0%** | −1.8% |
| `chat` | 716 | 1108 | 1118 | **+54.9%** | +0.9% |
| `high-concurrency` | 3725 | 3868 | 3867 | **+3.8%** | −0.0% |

On this GPU the search contributes nothing measurable once fp8 is chosen; all three deltas sit inside run-to-run noise of about ±2%. Choosing fp8 is still a decision stock vLLM does not make for you, and PolyServe makes it automatically and verifies it by measurement, but that is a narrower claim than "tuned configuration".

**Where the search does earn its keep: CPU.** With the GPU masked, the selector falls back to llama.cpp on CPU, and both sides run the same Q4_K_M weights, so quantisation is held constant:

| machine | PolyServe pick | tok/s | stock llama.cpp | tok/s | gain | TTFT |
|---|---|---|---|---|---|---|
| Intel i5-14600KF, CPU only, Qwen2.5-0.5B | Q4_K_M, ctx 8192, **8 slots** | **158** | Q4_K_M, ctx 4096, **1 slot** | 49 | **+224%** | 48 ms vs 642 ms |

Same backend, same quantisation, 3.2× the throughput and an SLO the default misses, purely from serving 8 requests concurrently instead of 1. This row is excluded from the headline above precisely because the baseline misses the latency target, which is the rule the report applies everywhere.

### Does fp8 cost quality?

Throughput across precisions is not like-for-like unless quality holds, so `benchmarks/quality_check.py` measures perplexity on identical held-out sequences (24 × 1024 tokens) through the same engine:

| precision | perplexity | change |
|---|---|---|
| bf16 | 5.2784 | baseline |
| fp8 | 5.3367 | **+1.10%** |

fp8 is not free: roughly 55% more throughput for about 1% worse perplexity on this model. Perplexity is a weak proxy for task quality; a task-level check (GSM8K, for instance) is still missing.

### Memory planner accuracy

Each trial compares the planner's memory estimate with actual allocations reported by NVML and the backend's startup log. Run `polyserve memory-report` to see the comparison, or add `--apply` to fit the planner's constants to your machine.

| backend | scored on | trials | mean abs error | bias | worst under-prediction | OOMs |
|---|---|---|---|---|---|---|
| vLLM | weights + workspace | 42 | **6.9%** | +3.1% | −12.9% | 0 |
| llama.cpp (CUDA) | peak device memory | 32 | **16.6%** | +16.6% | 0.0% | 0 |

Across 66 measured trials the planner predicts its target within **11.6%** on average, worst under-prediction −12.9%, and **zero out-of-memory failures among the configurations it admitted**.

*Scope, because the two files differ on purpose.* The table above is `polyserve memory-report`, which reads every calibration trial plus the comparison rows (66 trials). [benchmarks/RESULTS.md](benchmarks/RESULTS.md) reports the same metric over the comparison rows alone (16 trials), because that file must be reproducible from [benchmarks/results/](benchmarks/results/) by anyone who clones the repo. On the smaller subset the figures are 14.3% mean error and a −3.8% vLLM bias, and the KV-pool ratio falls to 1.1x because stock vLLM asks for 32k context at batch 256 and therefore budgets a pool nearly as large as the one it allocates. **The planner under-predicts vLLM's non-KV memory in the worst case by 12.9%, so `--apply` raises its safety margin from 5% to 14.9%** rather than lowering it; only llama.cpp's margin drops, to 2%. That last number is the one the planner is judged on: an admitted config that then OOMs is a planner failure regardless of average error.

The two backends are scored on different quantities on purpose. vLLM and SGLang size their KV pool to fill `gpu_memory_utilization × VRAM`, so their peak memory is a policy choice, not a requirement; scoring against it compares two different things. In this run vLLM's KV pool was **7.7× larger** than the planner budgeted, which is why the conservative `PAGED_KV_FRACTION` never rejected a workable config. llama.cpp allocates exactly what it is asked for, so it is scored on peak memory; its error is entirely over-prediction, traced to a hand-set 768 MB workspace constant against the 253 MB `polyserve memory-report --apply` fitted from these trials.

### Performance predictor

The predictor uses a roofline model with three parameters per backend. `polyserve fit` learns them from trials on the same machine. Accuracy is checked by leaving one trial out at a time and predicting its result.

| backend | observations | error, fitted | error, priors only | rank correlation (Spearman ρ) |
|---|---|---|---|---|
| vLLM | 90 | **22.1%** | 35.1% | **0.90** |
| llama.cpp (CUDA) | 78 | **23.4%** | 54.5% | **0.83** |

Fitting reduces the error by a third for vLLM and by more than half for llama.cpp. For narrowing the search, getting the ranking right matters more than predicting exact throughput. A rank correlation near 0.9 helps the predictor identify quantizations that are unlikely to win, saving a benchmark run.

The predictor still has limits. The fitted bandwidth and compute efficiencies reach their cap of 1.0, suggesting that the roofline model built on vendor peak numbers underestimates what these engines achieve. TTFT predictions are also much less accurate than throughput predictions: scheduling and queueing have a large effect, and the model only approximates them.

### Not yet measured

These are gaps, not claims. In rough order of how much they would change the conclusions:

1. **A task-level quality check.** Perplexity moved 1.1% at fp8, which is small, but perplexity is a weak proxy. GSM8K or a similar task-level benchmark at both precisions would say whether that 1.1% matters.
2. **A second model size.** Everything here is 3B on GPU and 0.5B on CPU. A 7B or 8B model on 24 GB is where the memory planner actually binds, and where the search may contribute more than it does at 3B.
3. **Other accelerators.** A100, A30 and a pre-Turing card (GTX 1080) are untested, so the compute-capability branch in the selector has never run on real hardware. SGLang is implemented and has never been benchmarked at all.
4. **Whether the search helps on GPU at all.** The isolation experiment says it does not, on one card with one model at three workloads. Finding out whether that holds on a card where memory is tight, or is an artifact of a 3B model on 24 GB, is the most interesting open question in this repository.
5. **Energy tuning on real hardware.** `--power` has only run against a simulated NVML. It needs root on the host, so it has to be measured on a machine you control. Whether the energy-optimal point sits near 70% of full power for these workloads, and whether it differs between prefill-heavy `rag` and decode-heavy `generation`, is still a prediction.
6. **Disaggregated prefill and decode on real GPUs.** `--phases disaggregated` has only run against fake engines. Whether moving the KV cache between two GPUs on one node beats a single engine at 3B, where the unified result already has memory headroom, is open, and a loss there would not be surprising: the published gains come from larger models and heavier prefill contention.

## Backend interface

To add support for new hardware, implement a backend class with the following interface. The core runtime stays the same:

```python
class Backend(Protocol):
    name: str
    def available(self, hw) -> bool
    def supports(self, hw, model) -> bool
    def prepare(self, model, hw) -> PreparedModel
    def materialize(self, model, quants) -> PreparedModel      # download / convert only what survived planning
    def memory_model(self, hw) -> MemoryModel                  # workspace constant + KV-token rule
    def estimate_memory(self, cfg, model, hw) -> int
    def candidate_configs(self, hw, model) -> list[Config]
    def launch(self, cfg, model, port) -> Process
    def workload_hooks(self, hw, model) -> LlmtraceHooks
```

v1 implementations: `VllmBackend`, `SglangBackend`, `LlamaCppCudaBackend`, `LlamaCppCpuBackend`, `VllmCpuBackend` in [polyserve/backends/](polyserve/backends/).

---

## Roadmap (not in v1)

- AMD / ROCm (vLLM-ROCm, llama.cpp HIP) — new `Backend` classes.
- Apple Silicon — MLX and llama.cpp Metal.
- Windows.
- Live re-tuning under real traffic instead of a one-time synthetic calibration.
- Custom kernels. PolyServe will keep sitting above the engines, not inside them.

## Non-goals

- **Not a replacement for vLLM / SGLang / llama.cpp.** It sits above them and launches them.
- **Not a compiler.** It is a benchmark-driven configuration planner.
- **Does not promise every machine.** The supported matrix above is explicit; anything else is a plugin.

## Development

```bash
pip install -e .[dev]
pytest            # all tests run without a GPU or any backend installed
```

For more detail on the memory planner, staged search and energy objective, see the [technical writeup](docs/writeup.md).

## License

MIT
