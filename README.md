# PolyServe

**PolyServe is a hardware-adaptive LLM serving runtime that automatically selects and tunes the best available inference backend for your machine.** One command, one OpenAI-compatible API, no backend or quantization configuration required.

```bash
pip install polyserve
polyserve serve meta-llama/Llama-3.2-3B-Instruct            # --objective balanced
curl localhost:8000/v1/chat/completions -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

First launch: discover hardware → prepare model → calibrate once → serve.
Later launches: load the cached profile → serve.

PolyServe is not a new inference engine. It sits above vLLM, SGLang and llama.cpp and answers the question every deployment starts with: *which backend, which quant, which memory / batch settings, on this box, for this objective?* It answers it by measuring, not by guessing.

---

## Supported matrix (v1)

| Hardware | Backends tried |
|---|---|
| NVIDIA, compute capability ≥ 7.5 (Turing and newer) | vLLM, SGLang, llama.cpp (CUDA) |
| NVIDIA, compute capability < 7.5 (Pascal, Volta) | llama.cpp (CUDA) |
| x86 CPU | llama.cpp; vLLM-CPU if AVX-512 is present |

Linux, Python 3.10–3.13. A backend is in v1 only if it gets a row in the benchmark table.

Each backend is a subprocess PolyServe launches with tuned arguments; you install the ones you want:

```bash
pip install polyserve[nvml]          # + NVML telemetry (power, utilisation) for calibration
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
    M --> PL[memory planner<br/>~144 configs → 9–15 feasible]
    PL --> C[calibrate<br/>staged search, llmtrace-measured]
    C --> CA[(profile cache<br/>~/.polyserve/profiles)]
    CA --> SV[serve<br/>supervised backend + OpenAI proxy :8000]
```

1. **Probe** — `HardwareDescriptor`: GPU vendor/name/compute capability/VRAM, CPU cores + AVX2/AVX-512, RAM, and which of vLLM / SGLang / llama.cpp / vLLM-CPU are actually runnable here. (`polyserve probe`)
2. **Select** — the rules in the table above. Multiple candidates are allowed; calibration picks the winner.
3. **Prepare** — vLLM/SGLang use HF weights as-is. llama.cpp resolves a pre-quantised GGUF from the Hub (Q4_K_M, Q5_K_M, Q6_K, Q8_0), or converts + quantises FP16 itself, and only for the quants the planner keeps.
4. **Memory planner** — before any process is launched:
   `estimated = weights + kv_cache(ctx, batch, dtype) + runtime_workspace + safety_margin`, keep a config only if `estimated ≤ 0.95 × available`. This prunes the grid from ~144 points to a dozen. (`polyserve plan <model>`)
5. **Calibrate** — a staged search, not a grid: (1) one short run per quant, keep the top two; (2) largest safe memory config; (3) batch / concurrency sweep. Each trial replays a fixed synthetic workload (16 prompts × 256-token prefill × 128-token decode at concurrency 1/4/8, ~10 s) and is measured with [llmtrace](https://github.com/Aagam-Bothara/llmtrace): tok/s, TTFT, TPOT, peak memory, GPU utilisation, power.
6. **Cache** — `~/.polyserve/profiles/<hardware_hash>/<model>/<objective>[-<workload>].json` holds the winner, the full launch args, the whole calibration table, and versions. Invalidated when the hardware or backend version changes; `polyserve recalibrate` forces a rerun.
7. **Serve** — the winner runs as a supervised subprocess (health check + auto-restart). A thin proxy on `:8000` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models` (streaming passthrough) and `/polyserve/profile`, which returns the active configuration and calibration table.

### Workloads

Real traffic is not one shape. `--workload` picks the synthetic workload every trial replays, and each preset carries its own TTFT ceiling for the `balanced` objective. Profiles are cached per workload, so `serve --workload rag` and `serve --workload chat` each get their own calibration.

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

All four are constrained argmax. There is no weighted score formula in v1.

| `--objective` | Rule |
|---|---|
| `throughput` | max tok/s |
| `latency` | min TTFT s.t. tok/s ≥ floor |
| `balanced` (default) | max tok/s s.t. TTFT ≤ ceiling (the workload's; `--ttft-ceiling` overrides) |
| `efficiency` | min joules/token s.t. tok/s ≥ floor |

The floor defaults to 50% of the best observed tok/s (`--tok-s-floor` for an absolute value). If nothing satisfies a constraint, the least-violating config wins and the profile says so.

---

## CLI

```
polyserve serve <model> [--objective X] [--workload W] [--port N] [--backend NAME] [--skip-calibration]
polyserve probe                 # print HardwareDescriptor
polyserve workloads             # list workload presets
polyserve plan <model>          # print feasible configs without running them
polyserve bench <model>         # run calibration and print the table, don't serve
polyserve recalibrate <model>   # force a rerun and overwrite the cached profile
polyserve profiles              # list cached profiles
```

`--skip-calibration` serves the first candidate backend with sane defaults immediately; nothing is cached.

---

## Benchmarks

Workload: 16 prompts × ~256-token prefill × 128-token decode, `--objective balanced` (TTFT ≤ 500 ms), measured with llmtrace (NVML at 100 ms). tok/s and TTFT are from the concurrency level the objective selected; W is mean device power during the trial; J/token is integrated device energy over output tokens.

### RTX 3090 (24 GB, cc 8.6), Qwen2.5-3B-Instruct, vLLM 0.11.0 + llama.cpp b-current

| Runtime / config | tok/s | TTFT p50 (ms) | peak mem (GB) | W | J/token |
|---|---|---|---|---|---|
| **PolyServe auto** → vLLM fp8, ctx 4096, max_num_seqs 64 | **1065** | 31 | 19.9 | 297 | **0.94** |
| vLLM bf16, same ctx / batch (stock precision) | 694 | 37 | 20.0 | 296 | 1.37 |
| PolyServe auto, llama.cpp only (`--backend llamacpp-cuda`) → Q4_K_M, ctx 8192, 8 slots, full offload | 645 | 72 | n/r | n/r | 0.44 |
| llama.cpp Q4_K_M, 4 slots, full offload (stage-1 baseline; Ollama-style defaults) | 545 | 952 | n/r | n/r | 0.80 |
| Ollama defaults | _not measured_ | | | | |

Calibration on this machine: 10 trials, 35 minutes with vLLM in the mix (vLLM startup with fp8 quantisation and CUDA-graph capture dominates; each trial's workload is ~10 s), 4 minutes for the llama.cpp-only run. n/r = not recorded in that run. The llama.cpp rows show the other kind of win: same quant, same offload, but 8 server slots instead of 4 turns a 952 ms queueing TTFT into 72 ms under an 8-client load. Peak memory for vLLM is its `gpu_memory_utilization` pre-allocation, not live usage. The fp8 pick is 1.5× the throughput and 31% less energy per token than bf16 on the same card, which is the kind of decision a default never makes for you.

Rows for the A100/A30 lab node and a GTX 1080 / CPU-only box land here from `polyserve bench` (week 6). Every _pending_ cell is a placeholder, not a claim.

## Backend interface

New hardware is a new class, no core changes:

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

Technical notes on the memory planner, staged search and energy objective: [docs/writeup.md](docs/writeup.md).

## License

MIT
