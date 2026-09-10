# PolyServe: a benchmark-driven configuration planner for LLM serving

*Technical notes — memory planner, staged search, energy objective.*

## 1. Problem

Serving one model on one machine means choosing a backend (vLLM, SGLang, llama.cpp, …), a precision or quant, a memory budget, a context length and a batch/concurrency limit. The space is small enough to enumerate (~144 points for one model on one box) and large enough that nobody does; people copy defaults, and defaults are tuned for the wrong GPU. PolyServe turns this into a measured decision that runs once per (machine, model, objective) and is cached.

The design has three parts that are reusable beyond this project: a **memory planner** that removes configurations that cannot run, a **staged search** that measures a dozen instead of a hundred, and a set of **constrained-argmax objectives** including an energy one, all measured by the same tracer (llmtrace).

## 2. Memory planner

Every configuration `cfg` is scored *before launch* with an additive model:

```
estimated(cfg) = weights(quant)
               + kv_cache(ctx, batch, kv_dtype)
               + runtime_workspace(backend)
               + safety_margin
keep cfg  iff  estimated(cfg) <= 0.95 * available
```

- `weights`: the GGUF file size reported by the Hub (or the converted file), or `params × bytes(dtype)` for HF weights. Parameter counts come from safetensors metadata when published, otherwise from a dense-transformer estimate (within 3% for Llama-3.2-3B).
- `kv_cache`: `2 × layers × kv_heads × head_dim × kv_tokens × bytes(kv_dtype)`. For multi-head attention this equals the textbook `2 × layers × hidden × …`; for GQA models it is the (smaller) correct number.
- `runtime_workspace`: one empirical constant per backend (vLLM 1.5 GiB, SGLang 2 GiB, llama.cpp 768 MiB / 512 MiB CPU). It covers CUDA context, CUDA graphs and activation buffers; it is the only fitted quantity in the model.
- `safety_margin`: `max(5% of available, 512 MB)`.
- `available`: *free* VRAM (or RAM) at probe time, not total. For vLLM/SGLang the budget is additionally capped at `gpu_memory_utilization × total`, because that is what the engine will refuse to exceed.

**KV tokens is backend-specific.** llama.cpp allocates the whole `-c` context up front, and `-c` is shared across `-np` slots, so `kv_tokens = ctx × n_parallel` exactly. Paged-KV runtimes (vLLM, SGLang) allocate a pool and preempt requests when it fills; requiring `ctx × max_num_seqs` tokens would reject every realistic batch on a 24 GB card. PolyServe budgets `0.25 × ctx × max_num_seqs` for them (`PAGED_KV_FRACTION`). This keeps the planner honest about what must fit (weights + workspace + a useful pool) without pretending the engine cannot run below worst-case occupancy.

**Partial offload.** For llama.cpp with `n_gpu_layers < layers`, the GPU estimate is scaled by the offloaded fraction and the remainder is checked against host RAM with its own margin. Both must fit.

Effect, computed by `polyserve plan` from the HF config alone (feasible / enumerated):

| machine | model | vLLM | SGLang | llama.cpp (CUDA) |
|---|---|---|---|---|
| A100 80 GB | Llama-3.2-3B | 52 / 54 | 26 / 27 | 108 / 108 |
| RTX 4090 24 GB | Llama-3.2-3B | 34 / 54 | 34 / 54 | 108 / 108 |
| RTX 4090 24 GB | Llama-3.1-8B | 22 / 54 | 20 / 54 | 108 / 108 |
| GTX 1080 8 GB | Llama-3.2-3B | not selected (cc 6.1) | not selected | 94 / 108 |
| GTX 1080 8 GB | Llama-3.1-8B | not selected | not selected | 46 / 108 |

A 3B model fits almost everywhere; the planner earns its keep as the model grows and the card shrinks. The llama.cpp grid on an 8 GB card for an 8B model loses every full-offload Q6/Q8 config and every 8-slot high-context one. The planner never launches a process, so this costs milliseconds; the staged search then decides how many of the survivors get measured.

## 3. Staged search

A full grid would be 144 launches × ~30 s (engine startup dominates, not the 10 s workload) ≈ 70 minutes. The staged search runs 8–14 trials:

1. **Quant / precision.** Group feasible configs by `(backend, quant)`. For each group run *one* baseline: the median config by (ctx, batch, memory knob). Rank by the objective; keep the top two groups. This is where backends get eliminated: a backend whose best quant is not in the top two is never launched again.
2. **Memory.** For each kept group, hold batch at the baseline and walk memory settings from largest to smallest — `gpu_memory_utilization` 0.95 → 0.90 → 0.80, or `n_gpu_layers` all → ¾ → ½, with ctx descending within each. Stop at the first config that launches *and* finishes the workload. That is "largest safe": the planner's estimate got it into the list, the measurement confirms it. At most three launches per group.
3. **Batch / concurrency.** With quant and memory fixed, sweep `max_num_seqs` (16 / 64 / 256) or `n_parallel` (1 / 4 / 8) and, on CPU, `n_batch`.

The final choice is the objective's constrained argmax over **every** successful trial from all three stages, not only stage 3, so a stage-1 baseline that happens to be best is not lost. Trials are keyed by config and never repeated. A trial that fails to launch (OOM, unsupported flag) is recorded in the table with its log tail; it is data, not an exception.

Every trial uses the same fixed workload — 16 prompts × ~256-token prefill × 128-token decode, `temperature 0`, `ignore_eos`, at concurrency 1, 4 and 8 — so numbers are comparable across backends and across machines. Summary metrics come from the concurrency level with the highest throughput (the load the server would actually be run at); energy is total joules over total tokens across all levels.

## 4. Measurement and the energy objective

Request timing (TTFT, TPOT, tok/s) is measured at the HTTP client, the only vantage point shared by all backends. Device telemetry comes from llmtrace's `GPUSampler` (NVML at 100 ms: power, utilisation, memory), falling back to bare `pynvml`, and on CPU to process RSS plus RAPL package energy when readable. Energy is the trapezoidal integral of sampled power over the trial window; joules/token divides it by output tokens.

The four objectives are all constrained argmax:

| objective | minimise / maximise | subject to |
|---|---|---|
| throughput | max tok/s | — |
| latency | min TTFT p50 | tok/s ≥ floor |
| balanced | max tok/s | TTFT p50 ≤ ceiling |
| efficiency | min J/token | tok/s ≥ floor |

The floor defaults to 50% of the best tok/s observed in the same calibration, so it is relative to the machine, not an absolute number that would be meaningless across an A100 and a laptop. The ceiling defaults to 500 ms. If no configuration satisfies the constraint, the least-violating one wins and the profile records that the constraint was relaxed. There is deliberately no weighted score: a weight is a hidden preference, a constraint is a stated one, and the calibration table is saved in full so the choice can be audited.

Why efficiency is its own objective rather than a tie-breaker: on the same GPU, J/token varies by 2–3× across batch sizes because idle power is amortised over more tokens at higher concurrency, while TTFT gets worse. `efficiency` picks the point on that curve that still meets a throughput floor. Whether that is the same point `balanced` picks is an empirical question the benchmark table answers per machine.

## 5. Caching and reproducibility

A profile is keyed by `(hardware_hash, model, objective)`. The hash covers the *shape* of the hardware (GPU model, compute capability, total VRAM, CPU model, core counts, SIMD flags, total RAM) and not free memory, so a busy machine does not invalidate its own cache. The profile stores the backend version; a version change invalidates it. It also stores the full calibration table, workload description and llmtrace version, so a decision can be re-derived offline.

## 6. Limitations

- One GPU. Tensor parallel is a stage the search does not have yet.
- The synthetic workload is a proxy. A deployment with 4k-token prompts or 2k-token outputs sits elsewhere on the throughput/latency curve. Live re-tuning under real traffic is on the roadmap.
- `runtime_workspace` is a constant per backend, not a function of model size; large models with many CUDA graphs will exceed it, which the stage-2 "largest safe" walk absorbs at the cost of one failed launch.
- Energy on CPU needs RAPL read permission; without it `efficiency` degrades to throughput ordering and says so.
