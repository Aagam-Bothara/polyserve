# PolyServe benchmark results

Across 6 GPU/model/workload combinations, PolyServe improves throughput by a median of +51% (range +7% to +67%) over the best stock/default configuration that satisfies the requested latency SLO, winning 6 of 6.

| machine | model | workload | PolyServe pick | tok/s | best default | tok/s | gain | TTFT Δ | J/tok gain | calib |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3090 | Qwen/Qwen2.5-3B-Instruct | chat | vllm/fp8/ctx8192/b64/gmu0.95 | 1091 | vllm-default | 653 | +67% | -3 ms | +42% | 399s / 10 |
| NVIDIA GeForce RTX 3090 | Qwen/Qwen2.5-3B-Instruct | default | vllm/fp8/ctx8192/b64/gmu0.95 | 1097 | vllm-default | 687 | +60% | -8 ms | +38% | 480s / 10 |
| NVIDIA GeForce RTX 3090 | Qwen/Qwen2.5-3B-Instruct | generation | vllm/fp8/ctx8192/b16/gmu0.95 | 1111 | vllm-default | 683 | +63% | -7 ms | +38% | 1937s / 10 |
| NVIDIA GeForce RTX 3090 | Qwen/Qwen2.5-3B-Instruct | high-concurrency | vllm/fp8/ctx8192/b64/gmu0.95 | 3628 | vllm-default | 3404 | +7% | -126 ms | +6% | 730s / 10 |
| NVIDIA GeForce RTX 3090 | Qwen/Qwen2.5-3B-Instruct | long-context | vllm/fp8/ctx32768/b16/gmu0.95 | 436 | vllm-default | 305 | +43% | -1 ms | +32% | 397s / 8 |
| NVIDIA GeForce RTX 3090 | Qwen/Qwen2.5-3B-Instruct | rag | vllm/fp8/ctx16384/b64/gmu0.95 | 705 | vllm-default | 504 | +40% | -23 ms | +30% | 370s / 8 |

⚠ = baseline or PolyServe missed the workload's TTFT SLO in that run; not counted in the headline.

## Memory planner accuracy

| backend | scored on | trials | measured | MAPE | bias | worst under | worst over | weights MAPE | workspace now → fitted (p95) | KV pool vs budget | OOMs | margin now → recommended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp-cuda | peak device | 6 | 6 | 23.1% | +23.1% | +0.0% | +23.4% | -% | 768 → 225 MB | - | 0 | 5.0% → 2.0% |
| vllm | weights+workspace | 12 | 10 | 9.0% | -3.8% | -12.9% | +12.3% | 6.0% | 1536 → 2588 MB | 1.1x | 0 | 5.0% → 14.9% |

Across 16 measured trials (18 planned) the planner predicts its target quantity within 14.3% on average; worst under-prediction -12.9%; 0 OOMs among planner-feasible configs.

Reservation backends (vLLM, SGLang) size their KV pool to fill `gpu_memory_utilization x VRAM`, so their peak is a policy choice, not a requirement: they are scored on weights + workspace, and the KV column shows how much larger the pool they allocated was than the planner budgeted. llama.cpp allocates exactly what it is asked for, so it is scored on peak device memory. Negative bias means the planner under-predicts; the recommended margin is the worst under-prediction plus 2%, clamped to [2%, 15%], with the 512 MB floor unchanged.


## Per-combination detail

**NVIDIA GeForce RTX 3090** · Qwen/Qwen2.5-3B-Instruct · workload `chat` · objective `balanced` (TTFT ≤ 500 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **PolyServe** → vllm/fp8/ctx8192/b64/gmu0.95 | 1091 @c8 | 37 | 39 | 7.0 | 23.4 GB | 304 | 0.268 | yes | 399s / 10 |
| vllm-default (vllm/bf16/ctx32768/b256/gmu0.90) | 653 @c8 | 40 | 54 | 11.9 | 22.7 GB | 306 | 0.463 | yes | - |
| llamacpp-cuda-default (llamacpp-cuda/Q4_K_M/ctx4096/b1/ngl37/nb2048) | 209 @c1 | 68 | 76 | 4.3 | 2.8 GB | 303 | 1.416 | yes | - |
| ollama-default (ollama/ollama/ctx0/b0) | 127 @c1 | 98 | 159 | 7.0 | 3.9 GB | 264 | 2.071 | yes | - |

**NVIDIA GeForce RTX 3090** · Qwen/Qwen2.5-3B-Instruct · workload `default` · objective `balanced` (TTFT ≤ 500 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **PolyServe** → vllm/fp8/ctx8192/b64/gmu0.95 | 1097 @c8 | 32 | 44 | 6.9 | 23.4 GB | 303 | 0.267 | yes | 480s / 10 |
| vllm-default (vllm/bf16/ctx32768/b256/gmu0.90) | 687 @c8 | 40 | 45 | 11.3 | 22.7 GB | 305 | 0.433 | yes | - |
| llamacpp-cuda-default (llamacpp-cuda/Q4_K_M/ctx4096/b1/ngl37/nb2048) | 213 @c1 | 51 | 54 | 4.3 | 2.8 GB | 302 | 1.377 | yes | - |
| ollama-default (ollama/ollama/ctx0/b0) | 131 @c1 | 48 | 79 | 7.2 | 3.9 GB | 260 | 1.949 | yes | - |

**NVIDIA GeForce RTX 3090** · Qwen/Qwen2.5-3B-Instruct · workload `generation` · objective `balanced` (TTFT ≤ 500 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **PolyServe** → vllm/fp8/ctx8192/b16/gmu0.95 | 1111 @c8 | 33 | 45 | 7.2 | 23.3 GB | 308 | 0.277 | yes | 1937s / 10 |
| vllm-default (vllm/bf16/ctx32768/b256/gmu0.90) | 683 @c8 | 40 | 51 | 11.7 | 22.7 GB | 309 | 0.450 | yes | - |
| llamacpp-cuda-default (llamacpp-cuda/Q4_K_M/ctx4096/b1/ngl37/nb2048) | 235 @c1 | 49 | 55 | 4.2 | 2.8 GB | 308 | 1.301 | yes | - |
| ollama-default (ollama/ollama/ctx0/b0) | 131 @c1 | 96 | 145 | 7.6 | 3.9 GB | 266 | 2.027 | yes | - |

**NVIDIA GeForce RTX 3090** · Qwen/Qwen2.5-3B-Instruct · workload `high-concurrency` · objective `balanced` (TTFT ≤ 1000 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **PolyServe** → vllm/fp8/ctx8192/b64/gmu0.95 | 3628 @c64 | 156 | 203 | 15.3 | 23.4 GB | 301 | 0.084 | yes | 730s / 10 |
| vllm-default (vllm/bf16/ctx32768/b256/gmu0.90) | 3404 @c128 | 282 | 1973 | 22.7 | 22.7 GB | 303 | 0.089 | yes | - |
| llamacpp-cuda-default (llamacpp-cuda/Q4_K_M/ctx4096/b1/ngl37/nb2048) | 211 @c32 | 9434 | 9466 | 4.2 | 2.8 GB | 308 | 1.453 | no | - |
| ollama-default (ollama/ollama/ctx0/b0) | 126 @c32 | 15761 | 15961 | 6.9 | 3.9 GB | 267 | 2.129 | no | - |

**NVIDIA GeForce RTX 3090** · Qwen/Qwen2.5-3B-Instruct · workload `long-context` · objective `balanced` (TTFT ≤ 2000 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **PolyServe** → vllm/fp8/ctx32768/b16/gmu0.95 | 436 @c4 | 75 | 98 | 8.8 | 23.4 GB | 306 | 0.684 | yes | 397s / 8 |
| vllm-default (vllm/bf16/ctx32768/b256/gmu0.90) | 305 @c4 | 77 | 90 | 12.8 | 22.7 GB | 307 | 1.003 | yes | - |
| llamacpp-cuda-default (llamacpp-cuda/Q4_K_M/ctx4096/b1/ngl37/nb2048) | failed: 24/24 requests failed | | | | | | | | |
| ollama-default (ollama/ollama/ctx0/b0) | 67 @c1 | 1339 | 1400 | 8.4 | 3.9 GB | 251 | 3.987 | yes | - |

**NVIDIA GeForce RTX 3090** · Qwen/Qwen2.5-3B-Instruct · workload `rag` · objective `balanced` (TTFT ≤ 1500 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **PolyServe** → vllm/fp8/ctx16384/b64/gmu0.95 | 705 @c8 | 104 | 126 | 9.5 | 23.4 GB | 303 | 0.410 | yes | 370s / 8 |
| vllm-default (vllm/bf16/ctx32768/b256/gmu0.90) | 504 @c8 | 127 | 132 | 14.0 | 22.7 GB | 304 | 0.584 | yes | - |
| llamacpp-cuda-default (llamacpp-cuda/Q4_K_M/ctx4096/b1/ngl37/nb2048) | failed: 48/48 requests failed | | | | | | | | |
| ollama-default (ollama/ollama/ctx0/b0) | 36 @c1 | 978 | 1071 | 8.1 | 3.9 GB | 235 | 6.656 | yes | - |
