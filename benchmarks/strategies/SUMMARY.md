## PolyServe against stock settings

| model | workload | PolyServe pick | tok/s | TTFT p50 | stock vLLM bf16 (gain) | stock vLLM fp8 (gain) | stock llama.cpp (gain) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | chat-system | `vllm/gptq/ctx16384/b256/gmu0.95` | 733 | 309 ms | 432 (+69%) | 548 (+34%) | 155 (+372%) |
| Qwen2.5-3B-Instruct | chat | `vllm/gptq/ctx4096/b64/gmu0.95` | 783 | 282 ms | 469 (+67%) | 575 (+36%) | 156 (+402%) |
| Qwen2.5-3B-Instruct | generation | `vllm/gptq/ctx8192/b256/gmu0.95` | 1157 | 98 ms | 547 (+111%) | 837 (+38%) | 188 (+514%) |
| Qwen2.5-3B-Instruct | high-concurrency | `vllm/awq/ctx8192/b256/gmu0.95` | 1786 | 778 ms | 1686 (+6%) | 1118 (+60%) | 165 (+981%), missed SLO |
| Qwen2.5-3B-Instruct | rag-shared | `vllm/awq/ctx32768/b64/gmu0.95` | 480 | 409 ms | 334 (+44%) | 378 (+27%) | failed |
| Qwen2.5-7B-Instruct | chat | `vllm/gptq/ctx8192/b256/gmu0.95` | 495 | 475 ms | 233 (+113%) | 203 (+144%) | 96 (+413%) |
| Qwen2.5-3B-Instruct | rag | `vllm/bf16/ctx32768/b64/gmu0.95` | 106 | 1288 ms | 105 (+1%) | 45 (+134%) | - |
| Qwen2.5-3B-Instruct | high-concurrency | `vllm/gptq/ctx8192/b256/gmu0.95 x2` | 3152 | 803 ms | 1715 (+84%) | 1346 (+134%) | - |
| Qwen2.5-3B-Instruct | chat-system | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64` | 455 | 171 ms | - | - | 157 (+190%) |

- Qwen2.5-3B-Instruct / high-concurrency: PolyServe serves 2 replicas on 2 GPUs; the stock rows use one GPU, so this row is not a like-for-like throughput comparison

## What each strategy is worth

**Qwen2.5-3B-Instruct / chat-system**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 432 | -40.8% | 255 ms | 16.4 ms | 2335 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 543 | -25.5% | 355 ms | 11.3 ms | 1790 ms | met |
| polyserve | `vllm/gptq/ctx16384/b256/gmu0.95` | 729 |  | 272 ms | 8.3 ms | 1328 ms | met |
| -int4 (fp8) | `-` | failed | | | | | does not fit in memory |
| +kv:fp8_e5m2 | `vllm/gptq/ctx16384/b256/gmu0.95/kvfp8_e5m2` | 714 | -2.1% | 280 ms | 8.5 ms | 1359 ms | met |
| -prefix | `vllm/gptq/ctx16384/b256/gmu0.95/nopc` | 318 | -56.3% | 426 ms | 9.1 ms | 1588 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx16384/b256/gmu0.95/sd=ngram:4` | 586 | -19.7% | 347 ms | 11.1 ms | 1751 ms | met |

**Qwen2.5-3B-Instruct / chat**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 454 | -40.9% | 204 ms | 15.8 ms | 2210 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 562 | -26.7% | 405 ms | 11.1 ms | 1817 ms | met |
| polyserve | `vllm/gptq/ctx4096/b64/gmu0.95` | 767 |  | 251 ms | 8.0 ms | 1268 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx4096/b64/gmu0.95` | 575 | -25.1% | 338 ms | 10.9 ms | 1721 ms | met |
| +kv:fp8_e5m2 | `vllm/gptq/ctx4096/b64/gmu0.95/kvfp8_e5m2` | 739 | -3.7% | 264 ms | 8.5 ms | 1341 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx4096/b64/gmu0.95/sd=ngram:4` | 714 | -6.9% | 232 ms | 8.9 ms | 1363 ms | met |

Speculative decoding `ngram:4` by concurrency (throughput stops gaining at c=1, latency at c=1):

| concurrency | tok/s off | tok/s on | request latency off | request latency on |
|---|---|---|---|---|
| 1 | 144 | 113 | 886 ms | 1134 ms |
| 4 | 485 | 360 | 1079 ms | 1446 ms |
| 16 | 1215 | 1015 | 1641 ms | 1998 ms |
| 64 | 1685 | 1625 | 4561 ms | 4739 ms |

**Qwen2.5-3B-Instruct / generation**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 543 | -53.6% | 95 ms | 14.6 ms | 15035 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 839 | -28.4% | 131 ms | 9.4 ms | 9724 ms | met |
| polyserve | `vllm/gptq/ctx8192/b256/gmu0.95` | 1171 |  | 108 ms | 6.7 ms | 6975 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx8192/b256/gmu0.95` | 836 | -28.6% | 130 ms | 9.4 ms | 9756 ms | met |
| +kv:fp8_e5m2 | `vllm/gptq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 1127 | -3.8% | 119 ms | 7.0 ms | 7242 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx8192/b256/gmu0.95/sd=ngram:4` | 1029 | -12.1% | 105 ms | 7.6 ms | 7929 ms | met |

**Qwen2.5-3B-Instruct / high-concurrency**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 1632 | -5.9% | 538 ms | 29.8 ms | 2417 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 1154 | -33.5% | 587 ms | 18.8 ms | 1770 ms | met |
| polyserve | `vllm/awq/ctx8192/b256/gmu0.95` | 1735 |  | 795 ms | 24.6 ms | 2344 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx8192/b256/gmu0.95` | 1309 | -24.6% | 855 ms | 32.8 ms | 2919 ms | met |
| +kv:fp8_e5m2 | `vllm/awq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 1656 | -4.5% | 881 ms | 24.4 ms | 2415 ms | met |
| +spec:ngram:4 | `vllm/awq/ctx8192/b256/gmu0.95/sd=ngram:4` | 1736 | +0.1% | 585 ms | 26.1 ms | 2229 ms | met |

**Qwen2.5-3B-Instruct / rag-shared**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 336 | -30.6% | 363 ms | 18.0 ms | 1494 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 374 | -22.7% | 505 ms | 13.4 ms | 1349 ms | met |
| polyserve | `vllm/awq/ctx32768/b64/gmu0.95` | 484 |  | 386 ms | 10.2 ms | 1031 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx32768/b64/gmu0.95` | 370 | -23.5% | 439 ms | 13.4 ms | 1283 ms | met |
| +kv:fp8_e5m2 | `vllm/awq/ctx32768/b64/gmu0.95/kvfp8_e5m2` | 486 | +0.5% | 404 ms | 9.8 ms | 1023 ms | met |
| -prefix | `vllm/awq/ctx32768/b64/gmu0.95/nopc` | 95 | -80.3% | 1377 ms | 20.7 ms | 2678 ms | met |
| +spec:ngram:4 | `vllm/awq/ctx32768/b64/gmu0.95/sd=ngram:4` | 438 | -9.4% | 396 ms | 11.8 ms | 1139 ms | met |

**Qwen2.5-7B-Instruct / chat**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 232 | -53.3% | 388 ms | 31.3 ms | 4363 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 205 | -58.9% | 483 ms | 16.5 ms | 2578 ms | met |
| polyserve | `vllm/gptq/ctx8192/b256/gmu0.95` | 497 |  | 468 ms | 11.9 ms | 1975 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx8192/b256/gmu0.95` | 204 | -59.0% | 480 ms | 16.6 ms | 2585 ms | met |
| +kv:fp8_e5m2 | `vllm/gptq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 490 | -1.5% | 480 ms | 12.1 ms | 2020 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx8192/b256/gmu0.95/sd=ngram:4` | 300 | -39.6% | 351 ms | 11.0 ms | 1752 ms | met |

**Qwen2.5-3B-Instruct / chat-system**

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| polyserve | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64` | 427 |  | 191 ms | 6.8 ms | 1053 ms | met |
| +kv:q8_0 | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/kvq8_0/sd=ngram:64` | 439 | +2.8% | 279 ms | 5.8 ms | 1016 ms | met |
| +kv:q4_0 | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/kvq4_0/sd=ngram:64` | 327 | -23.6% | 338 ms | 18.1 ms | 2637 ms | met |
| +prefix:cache_reuse | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64/cache_reuse=256` | 417 | -2.5% | 223 ms | 6.8 ms | 1090 ms | met |
| +prefix:kv_unified | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64/kv_unified=True` | 482 | +12.7% | 160 ms | 6.0 ms | 922 ms | met |
| -spec | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256` | 397 | -7.0% | 382 ms | 16.3 ms | 2448 ms | met |


## Quality

| model | weights | perplexity | vs bf16 | checkpoint |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | bf16 | 5.278 |  | Qwen/Qwen2.5-3B-Instruct |
| Qwen2.5-3B-Instruct | fp8 | 5.337 | +1.1% | Qwen/Qwen2.5-3B-Instruct |
| Qwen2.5-3B-Instruct | awq | 6.647 | +25.9% | Qwen/Qwen2.5-3B-Instruct-AWQ |
| Qwen2.5-3B-Instruct | gptq | 6.944 | +31.6% | Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 |
| Qwen2.5-7B-Instruct | bf16 | 2.290 |  | Qwen/Qwen2.5-7B-Instruct |
| Qwen2.5-7B-Instruct | fp8 | 2.327 | +1.6% | Qwen/Qwen2.5-7B-Instruct |
| Qwen2.5-7B-Instruct | awq | 3.043 | +32.9% | Qwen/Qwen2.5-7B-Instruct-AWQ |
| Qwen2.5-7B-Instruct | gptq | 3.124 | +36.4% | Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 |

| GGUF | perplexity | vs Q8_0 |
|---|---|---|
| Q8_0 | 4.293 |  |
| Q6_K | 4.332 | +0.9% |
| Q5_K_M | 4.454 | +3.8% |
| Q4_K_M | 4.726 | +10.1% |
