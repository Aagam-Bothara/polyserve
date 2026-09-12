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

1. **Quant / precision.** Group feasible configs by `(backend, quant)`. For each group run *one* baseline: median ctx, median batch, and the *largest* memory setting the planner accepted (full GPU offload, highest `gpu_memory_utilization`); if that launch fails, step down once or twice. Rank by the objective; keep the top two groups. This is where backends get eliminated: a backend whose best quant is not in the top two is never launched again. (An earlier draft used the median memory setting; on an RTX 3090 that ran llama.cpp with 27 of 36 layers on the GPU and measured 35 tok/s instead of 545. Quants must be compared at the memory setting they would actually be served at.)
2. **Memory.** For each kept group, hold batch at the baseline and walk memory settings from largest to smallest — `gpu_memory_utilization` 0.95 → 0.90 → 0.80, or `n_gpu_layers` all → ¾ → ½, with ctx descending within each. Stop at the first config that launches *and* finishes the workload. That is "largest safe": the planner's estimate got it into the list, the measurement confirms it. At most three launches per group.
3. **Batch / concurrency.** With quant and memory fixed, sweep `max_num_seqs` (16 / 64 / 256) or `n_parallel` (1 / 4 / 8) and, on CPU, `n_batch`.

The final choice is the objective's constrained argmax over **every** successful trial from all three stages, not only stage 3, so a stage-1 baseline that happens to be best is not lost. Trials are keyed by config and never repeated. A trial that fails to launch (OOM, unsupported flag) is recorded in the table with its log tail; it is data, not an exception.

Later stages vary the leader along one dimension at a time: the prefill budget, the KV-cache type, prefix-cache flags (on workloads whose prompts share a prefix), speculative decoding, and optionally power. The README's Search options section lists them.

Each trial is measured at concurrency 1, 4 and 8, and the objective scores it at whichever level best satisfies the constraint. For `balanced` that is the highest-throughput level whose TTFT is still under the ceiling, so the winner is a (config, load) pair the server can actually be run at, not a throughput number achieved with a TTFT the constraint forbids. Scores within 2% are treated as ties. A tie goes first to the configuration that switches on fewer optional strategies (a strategy is adopted only when it measurably wins), then to the larger context window, then the larger batch, then the lower energy: a 1% tok/s edge is measurement noise, a doubled context is a capability.

On an RTX 3090 with Qwen2.5-3B this produced two decisions a default would not make: vLLM fp8 over bf16 (1065 vs 694 tok/s, 0.94 vs 1.37 J/token), and llama.cpp with 8 parallel slots instead of 4 (645 vs 545 tok/s, TTFT 72 ms vs 952 ms under the same 8-client load, because with 4 slots half the clients queue). Those numbers came from the harness that replayed prompts across concurrency levels (see below), so their absolute values are inflated. Re-measured with fresh prompts on an A40, fp8 still beat bf16 on `chat` (592 vs 464 tok/s) but lost to it on the prefill-heavy `high-concurrency` workload (1358 vs 1724), where fp8's weight-only kernels on Ampere cost more in compute-bound prefill than they save in bandwidth. The llama.cpp slot decision has not been re-measured.

Every trial uses the same fixed workload — 16 prompts × ~256-token prefill × 128-token decode, `temperature 0`, `ignore_eos`, at concurrency 1, 4 and 8 — so numbers are comparable across backends and across machines. Each concurrency level gets fresh prompts. Replaying one level's prompts at the next lets the engine's prefix cache make every later prefill nearly free: an early version of the harness did that and read one configuration at 3638 tok/s that measures 1814 with fresh prompts. A workload's deliberately shared prefix (the `chat-system` and `rag-shared` presets) is kept across levels, since caching it is the behaviour being measured. Summary metrics come from the concurrency level with the highest throughput (the load the server would actually be run at); energy is total joules over total tokens across all levels.

## 4. Measurement and the energy objective

Request timing (TTFT, TPOT, tok/s) is measured at the HTTP client, the only vantage point shared by all backends. Device telemetry comes from llmtrace's `GPUSampler` (NVML at 100 ms: power, utilisation, memory), falling back to bare `pynvml`, and on CPU to process RSS plus RAPL package energy when readable. Energy is the trapezoidal integral of sampled power over the trial window; joules/token divides it by output tokens.

The four objectives are all constrained argmax:

| objective | minimise / maximise | subject to |
|---|---|---|
| throughput | max tok/s | — |
| latency | min request latency (TTFT p50 + TPOT × output tokens) | tok/s ≥ floor |
| balanced | max tok/s | TTFT p50 ≤ ceiling |
| efficiency | min J/token | tok/s ≥ floor |

The floor defaults to 50% of the best tok/s observed in the same calibration, so it is relative to the machine, not an absolute number that would be meaningless across an A100 and a laptop. The ceiling defaults to 500 ms. If no configuration satisfies the constraint, the least-violating one wins and the profile records that the constraint was relaxed. There is deliberately no weighted score: a weight is a hidden preference, a constraint is a stated one, and the calibration table is saved in full so the choice can be audited.

Why efficiency is its own objective rather than a tie-breaker: on the same GPU, J/token varies by 2–3× across batch sizes because idle power is amortised over more tokens at higher concurrency, while TTFT gets worse. `efficiency` picks the point on that curve that still meets a throughput floor. Whether that is the same point `balanced` picks is an empirical question the benchmark table answers per machine.

## 5. Calibrating the planner against itself

The memory planner is a formula; the question is how wrong it is. Every trial now records the
planner's estimate (weights + KV + workspace, before the safety margin) next to two measurements:
the NVML peak during the trial minus what the device held before launch, and the backend's own
accounting parsed from its startup log (vLLM prints weights, non-KV total and available KV cache;
llama-server prints model, KV and compute buffer sizes per device; SGLang prints weight and KV
pool sizes). `polyserve memory-report` reduces these to, per backend: mean absolute % error,
signed bias, the worst under-prediction, component-wise error for weights and KV, and the number of
OOMs among planner-feasible configs. That last number is the one the planner is judged on: a
config the planner admitted that then OOMed is a planner failure, whatever the average error says.

Two constants are then fitted rather than assumed. `runtime_workspace`, the only hand-set term,
becomes the 95th percentile of measured workspace on this machine. The safety margin becomes the
worst observed under-prediction plus 2%, clamped to [2%, 15%], with the 512 MB floor kept.
`--apply` writes both to `~/.polyserve/memory-model.json` keyed by hardware hash and the planner
uses them from then on. The claim the matrix is meant to support is therefore checkable:
"predicts peak device memory within X% across N GPUs, with zero OOMs among the configs it admitted".

## 6. A performance predictor for the search

Measuring is the ground truth, but a search that must launch an engine to learn anything is slow
(vLLM start-up dominates at two to three minutes a trial). A predictor that ranks configs before
launch lets the search skip what cannot win. The model is a roofline, not a learned black box:

```
decode step, n sequences in flight:
    t_mem  = (device_weights + n · kv_bytes/token · (L_p + L_d/2)) / (α · mem_bw)
    t_comp = n · 2 · params / (β · peak_flops)
    t_step = max(t_mem, t_comp) + overhead
prefill (compute bound):  t_pre = n · L_p · 2 · params / (β · peak_flops) + device_weights / (α · mem_bw)
wave:   t_wave = t_pre + L_d · t_step        tok/s = n · L_d / t_wave
own prefill (chunked prefill interleaves requests): t_own = L_p · 2 · params / (β · peak_flops)
TTFT p50 with c clients and B slots (n = min(c, B)):  t_own + ⌊(c/2)/n⌋ · t_wave
```

Device peak bandwidth and FLOPS come from a small table keyed on the NVML name; the three
parameters per backend (α bandwidth efficiency, β compute efficiency, per-step overhead) are fitted
by grid search on log error from the machine's own calibration trials, one observation per
(config, concurrency level). `polyserve fit` reports leave-one-trial-out MAPE for tok/s and TTFT
and the Spearman rank correlation between predicted and measured tok/s, which is the number that
matters: pruning is safe when the ranking is right even if the magnitudes are off.

The search uses it conservatively. With fitted parameters, stage 1 skips a (backend, quant) group
whose predicted best is below 40% of the best predicted group, and stage 3 tries batch settings in
predicted order so an interrupted calibration already holds the likely winner. With priors only,
nothing is pruned; every prediction carries a fitted/prior flag. The queueing term is what made
the llama.cpp result on the RTX 3090 explicable before it was measured: with 8 clients on 4 slots
the median request waits a full wave, which is the 952 ms TTFT the trial recorded, and 8 slots
remove the wait.

## 7. Caching and reproducibility

A profile is keyed by `(hardware_hash, model, objective)`. The hash covers the *shape* of the hardware (GPU model, compute capability, total VRAM, CPU model, core counts, SIMD flags, total RAM) and not free memory, so a busy machine does not invalidate its own cache. The profile stores the backend version; a version change invalidates it. It also stores the full calibration table, workload description and llmtrace version, so a decision can be re-derived offline.

## 8. Limitations

- One GPU. Tensor parallel is a stage the search does not have yet.
- The synthetic workload is a proxy. A deployment with 4k-token prompts or 2k-token outputs sits elsewhere on the throughput/latency curve. Live re-tuning under real traffic is on the roadmap.
- `runtime_workspace` is a constant per backend, not a function of model size; large models with many CUDA graphs will exceed it, which the stage-2 "largest safe" walk absorbs at the cost of one failed launch.
- Energy on CPU needs RAPL read permission; without it `efficiency` degrades to throughput ordering and says so.
