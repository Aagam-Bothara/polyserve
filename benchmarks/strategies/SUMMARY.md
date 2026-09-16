## PolyServe against stock settings

| GPU | model | workload | PolyServe pick | tok/s | TTFT p50 | stock vLLM bf16 (gain) | stock vLLM fp8 (gain) | stock llama.cpp (gain) | stock SGLang (gain) |
|---|---|---|---|---|---|---|---|---|---|
| A40 | Qwen2.5-3B-Instruct | chat-system | `vllm/gptq/ctx16384/b256/gmu0.95` | 733 | 309 ms | 432 (+69%) | 548 (+34%) | 155 (+372%) | - |
| A40 | Qwen2.5-3B-Instruct | chat | `vllm/gptq/ctx4096/b64/gmu0.95` | 783 | 282 ms | 469 (+67%) | 575 (+36%) | 156 (+402%) | - |
| A40 | Qwen2.5-3B-Instruct | generation | `vllm/gptq/ctx8192/b256/gmu0.95` | 1157 | 98 ms | 547 (+111%) | 837 (+38%) | 188 (+514%) | - |
| A40 | Qwen2.5-3B-Instruct | high-concurrency | `vllm/awq/ctx8192/b256/gmu0.95` | 1786 | 778 ms | 1686 (+6%) | 1118 (+60%) | 165 (+981%), missed SLO | - |
| A40 | Qwen2.5-3B-Instruct | rag-shared | `vllm/awq/ctx32768/b64/gmu0.95` | 480 | 409 ms | 334 (+44%) | 378 (+27%) | failed | - |
| A40 | Qwen2.5-7B-Instruct | chat | `vllm/gptq/ctx8192/b256/gmu0.95` | 495 | 475 ms | 233 (+113%) | 203 (+144%) | 96 (+413%) | - |
| CPU | Qwen2.5-0.5B-Instruct | default | `llamacpp-cpu/Q4_K_M/ctx8192/b1/ngl0/nb2048/sd=ngram:64` | 92 | 475 ms | - | - | 68 (+37%) | - |
| A40 | Qwen2.5-3B-Instruct | rag | `vllm/bf16/ctx32768/b64/gmu0.95` | 106 | 1288 ms | 105 (+1%) | 45 (+134%) | - | - |
| A40 | Qwen2.5-3B-Instruct | file-dolly-d03d5896 | `vllm/bf16/ctx16384/b64/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 547 | 120 ms | 496 (+10%) | - | - | 473 (+16%) |
| A40 | Qwen2.5-3B-Instruct | file-dolly-d03d5896 (budget-10m) | `vllm/bf16/ctx16384/b64/gmu0.95` | 504 | 47 ms | 506 (-0%) | - | - | 474 (+6%) |
| A40 | Qwen2.5-3B-Instruct | file-dolly-d03d5896 (p95) | `vllm/bf16/ctx16384/b64/gmu0.95/pb16384/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 619 | 107 ms | 504 (+23%) | - | - | 475 (+30%) |
| A40 | Qwen2.5-3B-Instruct | file-dolly-d03d5896 (p95-old-warmup) | `vllm/bf16/ctx8192/b64/gmu0.95/kvfp8_e5m2` | 523 | 55 ms | 495 (+6%) | - | - | 492 (+6%) |
| A40 | Qwen2.5-3B-Instruct | file-dolly-d03d5896 (p95-old-warmup-budget) | `vllm/bf16/ctx8192/b64/gmu0.95` | 499 | 64 ms | 509 (-2%) | - | - | 475 (+5%) |
| L4 | Qwen2.5-7B-Instruct | chat | `vllm/fp8/ctx8192/b64/gmu0.95` | 195 | 487 ms | 17 (+1036%), missed SLO | 195 (+0%) | 49 (+301%) | - |
| L4 | Qwen2.5-7B-Instruct | sharegpt | `vllm/fp8/ctx8192/b64/gmu0.95/pb16384/kvfp8` | 738 | 985 ms | 410 (+80%) | 701 (+5%) | failed | - |
| L4 | Qwen2.5-7B-Instruct | sharegpt (p95) | `vllm/fp8/ctx8192/b64/gmu0.95/pb8192` | 219 | 176 ms | 128 (+71%) | 212 (+3%) | - | - |
| A40 | Qwen2.5-3B-Instruct | high-concurrency | `vllm/gptq/ctx8192/b256/gmu0.95 x2` | 3152 | 803 ms | 1715 (+84%) | 1346 (+134%) | - | - |
| A100-SXM4-80GB | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (baselines) | `vllm/bf16/ctx16384/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 1150 | 74 ms | 724 (+59%) | - | - | - |
| A40 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (baselines) | `vllm/bf16/ctx8192/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 505 | 168 ms | 262 (+93%) | - | - | - |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (baselines) | `sglang/fp8/ctx16384/b16/gmu0.93` | 763 | 53 ms | - | - | - | 448 (+70%) |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (explore) | `vllm/fp8/ctx16384/b16/gmu0.95/sd=suffix:24` | 815 | 30 ms | - | - | - | 448 (+82%) |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (headtohead) | `vllm/fp8/ctx16384/b16/gmu0.95/sd=suffix:24` | 813 | 30 ms | - | - | - | - |
| H100 NVL | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (heldout) | `vllm/fp8/ctx16384/b64/gmu0.95/pb8192/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 2164 | 44 ms | 1155 (+87%) | 1743 (+24%) | - | - |
| A100-SXM4-80GB | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (heldout) | `vllm/bf16/ctx16384/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 1153 | 74 ms | 724 (+59%) | - | - | 722 (+60%) |
| A40 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (heldout) | `vllm/bf16/ctx8192/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 505 | 167 ms | 262 (+93%) | - | - | 257 (+97%) |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (heldout) | `sglang/fp8/ctx16384/b16/gmu0.93` | 765 | 52 ms | failed | failed | - | 448 (+71%) |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (heldout-fixed) | `vllm/fp8/ctx16384/b16/gmu0.95/sd=suffix:24` | 807 | 31 ms | - | - | - | 459 (+76%) |
| H100 NVL | Meta-Llama-3.1-8B-Instruct | file-oasst-8106683e (oasst) | `vllm/fp8/ctx16384/b64/gmu0.95/pb8192/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 1902 | 48 ms | - | 1755 (+8%) | - | - |
| A100-SXM4-80GB | Meta-Llama-3.1-8B-Instruct | file-oasst-8106683e (oasst) | `vllm/bf16/ctx16384/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 1054 | 74 ms | 717 (+47%) | - | - | 716 (+47%) |
| A40 | Meta-Llama-3.1-8B-Instruct | file-oasst-8106683e (oasst) | `vllm/bf16/ctx8192/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 440 | 166 ms | 260 (+69%) | - | - | 264 (+67%) |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-oasst-8106683e (oasst) | `sglang/fp8/ctx16384/b16/gmu0.93` | 775 | 51 ms | - | failed | - | 460 (+68%) |
| A40 | Qwen2.5-3B-Instruct | file-dolly-heldout-fc5b6407 (qwen-heldout) | `vllm/bf16/ctx16384/b64/gmu0.95/pb8192/kvfp8_e5m2/sd=suffix:24` | 994 | 39 ms | 448 (+122%) | - | - | 472 (+110%) |
| A100-SXM4-80GB | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (specfirst) | `vllm/bf16/ctx16384/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 1155 | 74 ms | 726 (+59%) | - | - | - |
| A40 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (specfirst) | `vllm/bf16/ctx8192/b64/gmu0.95/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 506 | 167 ms | 262 (+93%) | - | - | - |
| GeForce RTX 4090 | Meta-Llama-3.1-8B-Instruct | file-dolly-heldout-fc5b6407 (specfirst) | `vllm/fp8/ctx16384/b16/gmu0.95/pb2048/kvint8_per_token_head/sd=draft:unsloth/Llama-3.2-1B-Instruct:4` | 1041 | 88 ms | - | - | - | 448 (+133%) |
| A40 | Qwen2.5-3B-Instruct | chat-system | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64` | 455 | 171 ms | - | - | 157 (+190%) | - |
| A40 | Qwen2.5-3B-Instruct | code-edit | `vllm/bf16/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 483 | 64 ms | 459 (+5%) | - | 187 (+158%) | - |
| A40 | Qwen2.5-3B-Instruct | extract | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 628 | 177 ms | 439 (+43%) | - | 175 (+259%) | - |
| A40 | Qwen2.5-3B-Instruct | sharegpt | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2` | 1891 | 380 ms | 1714 (+10%) | - | 190 (+897%) | - |
| A40 | Qwen2.5-3B-Instruct | sharegpt (sglang) | `sglang/bf16/ctx8192/b64/gmu0.93/kvfp8_e5m2` | 1952 | 364 ms | - | - | - | 1799 (+8%) |
| A40 | Qwen2.5-3B-Instruct | extract (v2) | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 638 | 178 ms | 439 (+45%) | - | 176 (+261%) | - |
| A40 | Qwen2.5-3B-Instruct | extract (v3) | `vllm/bf16/ctx8192/b256/gmu0.95/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 731 | 171 ms | 439 (+66%) | - | 174 (+319%) | - |

- Qwen2.5-3B-Instruct / high-concurrency: PolyServe serves 2 replicas on 2 GPUs; the stock rows use one GPU, so this row is not a like-for-like throughput comparison

## What each strategy is worth

**Qwen2.5-3B-Instruct / chat-system** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 432 | -40.8% | 255 ms | 16.4 ms | 2335 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 543 | -25.5% | 355 ms | 11.3 ms | 1790 ms | met |
| polyserve | `vllm/gptq/ctx16384/b256/gmu0.95` | 729 |  | 272 ms | 8.3 ms | 1328 ms | met |
| -int4 (fp8) | `-` | failed | | | | | does not fit in memory |
| +kv:fp8_e5m2 | `vllm/gptq/ctx16384/b256/gmu0.95/kvfp8_e5m2` | 714 | -2.1% | 280 ms | 8.5 ms | 1359 ms | met |
| -prefix | `vllm/gptq/ctx16384/b256/gmu0.95/nopc` | 318 | -56.3% | 426 ms | 9.1 ms | 1588 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx16384/b256/gmu0.95/sd=ngram:4` | 586 | -19.7% | 347 ms | 11.1 ms | 1751 ms | met |

**Qwen2.5-3B-Instruct / chat** (A40)

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

**Qwen2.5-3B-Instruct / generation** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 543 | -53.6% | 95 ms | 14.6 ms | 15035 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 839 | -28.4% | 131 ms | 9.4 ms | 9724 ms | met |
| polyserve | `vllm/gptq/ctx8192/b256/gmu0.95` | 1171 |  | 108 ms | 6.7 ms | 6975 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx8192/b256/gmu0.95` | 836 | -28.6% | 130 ms | 9.4 ms | 9756 ms | met |
| +kv:fp8_e5m2 | `vllm/gptq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 1127 | -3.8% | 119 ms | 7.0 ms | 7242 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx8192/b256/gmu0.95/sd=ngram:4` | 1029 | -12.1% | 105 ms | 7.6 ms | 7929 ms | met |

**Qwen2.5-3B-Instruct / high-concurrency** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 1632 | -5.9% | 538 ms | 29.8 ms | 2417 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 1154 | -33.5% | 587 ms | 18.8 ms | 1770 ms | met |
| polyserve | `vllm/awq/ctx8192/b256/gmu0.95` | 1735 |  | 795 ms | 24.6 ms | 2344 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx8192/b256/gmu0.95` | 1309 | -24.6% | 855 ms | 32.8 ms | 2919 ms | met |
| +kv:fp8_e5m2 | `vllm/awq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 1656 | -4.5% | 881 ms | 24.4 ms | 2415 ms | met |
| +spec:ngram:4 | `vllm/awq/ctx8192/b256/gmu0.95/sd=ngram:4` | 1736 | +0.1% | 585 ms | 26.1 ms | 2229 ms | met |

**Qwen2.5-3B-Instruct / rag-shared** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 336 | -30.6% | 363 ms | 18.0 ms | 1494 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 374 | -22.7% | 505 ms | 13.4 ms | 1349 ms | met |
| polyserve | `vllm/awq/ctx32768/b64/gmu0.95` | 484 |  | 386 ms | 10.2 ms | 1031 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx32768/b64/gmu0.95` | 370 | -23.5% | 439 ms | 13.4 ms | 1283 ms | met |
| +kv:fp8_e5m2 | `vllm/awq/ctx32768/b64/gmu0.95/kvfp8_e5m2` | 486 | +0.5% | 404 ms | 9.8 ms | 1023 ms | met |
| -prefix | `vllm/awq/ctx32768/b64/gmu0.95/nopc` | 95 | -80.3% | 1377 ms | 20.7 ms | 2678 ms | met |
| +spec:ngram:4 | `vllm/awq/ctx32768/b64/gmu0.95/sd=ngram:4` | 438 | -9.4% | 396 ms | 11.8 ms | 1139 ms | met |

**Qwen2.5-7B-Instruct / chat** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 232 | -53.3% | 388 ms | 31.3 ms | 4363 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 205 | -58.9% | 483 ms | 16.5 ms | 2578 ms | met |
| polyserve | `vllm/gptq/ctx8192/b256/gmu0.95` | 497 |  | 468 ms | 11.9 ms | 1975 ms | met |
| -int4 (fp8) | `vllm/fp8/ctx8192/b256/gmu0.95` | 204 | -59.0% | 480 ms | 16.6 ms | 2585 ms | met |
| +kv:fp8_e5m2 | `vllm/gptq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 490 | -1.5% | 480 ms | 12.1 ms | 2020 ms | met |
| +spec:ngram:4 | `vllm/gptq/ctx8192/b256/gmu0.95/sd=ngram:4` | 300 | -39.6% | 351 ms | 11.0 ms | 1752 ms | met |

**Qwen2.5-3B-Instruct / file-dolly-d03d5896** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| polyserve | `vllm/bf16/ctx16384/b64/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 571 |  | 119 ms | 14.2 ms | 6173 ms | met |
| +awq | `vllm/awq/ctx16384/b64/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 625 | +9.4% | 100 ms | 11.3 ms | 4630 ms | met |
| +gptq | `vllm/gptq/ctx16384/b64/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 712 | +24.6% | 98 ms | 10.8 ms | 4699 ms | met |
| -kv | `vllm/bf16/ctx16384/b64/gmu0.95/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 565 | -1.0% | 98 ms | 10.5 ms | 2652 ms | met |
| -kv (FlashInfer kept) | `vllm/bf16/ctx16384/b64/gmu0.95/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4/attention_backend=FLASHINFER` | 541 | -5.3% | 104 ms | 11.5 ms | 3058 ms | met |
| -spec | `vllm/bf16/ctx16384/b64/gmu0.95/kvfp8_e5m2` | 509 | -10.8% | 63 ms | 13.7 ms | 5221 ms | met |

**Qwen2.5-7B-Instruct / chat** (L4)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 17 | -91.2% | 157 ms | 57.5 ms | 7460 ms | missed |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 194 | -0.0% | 491 ms | 37.2 ms | 5216 ms | met |
| polyserve | `vllm/fp8/ctx8192/b64/gmu0.95` | 194 |  | 489 ms | 37.2 ms | 5214 ms | met |
| +awq | `vllm/awq/ctx8192/b64/gmu0.95` | 48 | -75.1% | 156 ms | 19.6 ms | 2640 ms | met |
| +gptq | `vllm/gptq/ctx8192/b64/gmu0.95` | 49 | -74.8% | 153 ms | 19.4 ms | 2614 ms | met |
| +kv:fp8 | `vllm/fp8/ctx8192/b64/gmu0.95/kvfp8` | 197 | +1.3% | 488 ms | 36.7 ms | 5144 ms | met |
| +spec:ngram:4 | `vllm/fp8/ctx8192/b64/gmu0.95/sd=ngram:4` | 105 | -45.8% | 290 ms | 35.9 ms | 4851 ms | met |

**Qwen2.5-7B-Instruct / sharegpt** (L4)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 410 | -44.2% | 632 ms | 69.3 ms | 32710 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | 698 | -5.0% | 634 ms | 40.7 ms | 19540 ms | met |
| polyserve | `vllm/fp8/ctx8192/b64/gmu0.95/pb16384/kvfp8` | 734 |  | 999 ms | 43.7 ms | 21925 ms | met |
| +awq | `vllm/awq/ctx8192/b64/gmu0.95/pb16384/kvfp8` | 373 | -49.2% | 449 ms | 20.4 ms | 10893 ms | met |
| +gptq | `vllm/gptq/ctx8192/b64/gmu0.95/pb16384/kvfp8` | 376 | -48.8% | 443 ms | 20.3 ms | 10806 ms | met |
| -kv | `vllm/fp8/ctx8192/b64/gmu0.95/pb16384` | 681 | -7.3% | 836 ms | 40.2 ms | 19045 ms | met |
| +spec:ngram:4 | `vllm/fp8/ctx8192/b64/gmu0.95/pb16384/kvfp8/sd=ngram:4` | 725 | -1.2% | 965 ms | 43.1 ms | 21590 ms | met |

**Qwen2.5-3B-Instruct / chat-system** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| polyserve | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64` | 427 |  | 191 ms | 6.8 ms | 1053 ms | met |
| +kv:q8_0 | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/kvq8_0/sd=ngram:64` | 439 | +2.8% | 279 ms | 5.8 ms | 1016 ms | met |
| +kv:q4_0 | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/kvq4_0/sd=ngram:64` | 327 | -23.6% | 338 ms | 18.1 ms | 2637 ms | met |
| +prefix:cache_reuse | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64/cache_reuse=256` | 417 | -2.5% | 223 ms | 6.8 ms | 1090 ms | met |
| +prefix:kv_unified | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256/sd=ngram:64/kv_unified=True` | 482 | +12.7% | 160 ms | 6.0 ms | 922 ms | met |
| -spec | `llamacpp-cuda/Q5_K_M/ctx16384/b8/ngl37/nb512/pb256` | 397 | -7.0% | 382 ms | 16.3 ms | 2448 ms | met |

**Qwen2.5-3B-Instruct / code-edit** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 472 | -8.8% | 60 ms | 13.6 ms | 3175 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | failed | | | | | failed to start (rc=1) |
| polyserve | `vllm/bf16/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 517 |  | 67 ms | 13.8 ms | 5051 ms | met |
| +awq | `vllm/awq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 1121 | +116.6% | 49 ms | 6.0 ms | 2201 ms | met |
| +gptq | `vllm/gptq/ctx8192/b256/gmu0.95/kvfp8_e5m2` | 1150 | +122.2% | 68 ms | 6.0 ms | 2459 ms | met |
| -kv | `vllm/bf16/ctx8192/b256/gmu0.95` | 460 | -11.1% | 61 ms | 13.6 ms | 3094 ms | met |
| -kv (FlashInfer kept) | `vllm/bf16/ctx8192/b256/gmu0.95/attention_backend=FLASHINFER` | 485 | -6.3% | 62 ms | 13.8 ms | 3193 ms | met |
| +spec:ngram:4 | `vllm/bf16/ctx8192/b256/gmu0.95/kvfp8_e5m2/sd=ngram:4` | 143 | -72.3% | 40 ms | 7.2 ms | 2643 ms | met |
| +spec:draft:Qwen/Qwen2.5-0.5B-Instruct:4 | `vllm/bf16/ctx8192/b256/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 462 | -10.7% | 123 ms | 15.5 ms | 5887 ms | met |

**Qwen2.5-3B-Instruct / extract** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 439 | -27.9% | 109 ms | 14.8 ms | 3557 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | failed | | | | | failed to start (rc=1) |
| polyserve | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 609 |  | 189 ms | 11.7 ms | 4669 ms | met |
| +awq | `vllm/awq/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 728 | +19.5% | 174 ms | 8.9 ms | 3504 ms | met |
| +gptq | `vllm/gptq/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 747 | +22.6% | 195 ms | 8.9 ms | 3616 ms | met |
| -kv | `vllm/bf16/ctx8192/b256/gmu0.95/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 728 | +19.5% | 165 ms | 9.4 ms | 2378 ms | met |
| -kv (FlashInfer kept) | `vllm/bf16/ctx8192/b256/gmu0.95/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4/attention_backend=FLASHINFER` | 702 | +15.3% | 174 ms | 9.6 ms | 2229 ms | met |
| -spec | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2` | 537 | -11.9% | 276 ms | 14.1 ms | 5678 ms | met |

**Qwen2.5-3B-Instruct / sharegpt** (A40)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| vllm-default | `vllm/bf16/ctx32768/b256/gmu0.90` | 1680 | -7.8% | 251 ms | 15.7 ms | 7029 ms | met |
| vllm-fp8-default | `vllm/fp8/ctx32768/b256/gmu0.90` | failed | | | | | failed to start (rc=1) |
| polyserve | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2` | 1822 |  | 369 ms | 15.6 ms | 7703 ms | met |
| +awq | `vllm/awq/ctx8192/b512/gmu0.95/kvfp8_e5m2` | 3818 | +109.6% | 336 ms | 7.4 ms | 3929 ms | met |
| +gptq | `vllm/gptq/ctx8192/b512/gmu0.95/kvfp8_e5m2` | 3817 | +109.5% | 332 ms | 7.6 ms | 3998 ms | met |
| -kv | `vllm/bf16/ctx8192/b256/gmu0.95` | 1676 | -8.0% | 247 ms | 15.6 ms | 6945 ms | met |
| +spec:ngram:4 | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=ngram:4` | 244 | -86.6% | 350 ms | 86.2 ms | 42941 ms | met |
| +spec:draft:Qwen/Qwen2.5-0.5B-Instruct:4 | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4` | 1323 | -27.4% | 283 ms | 17.0 ms | 8586 ms | met |

**Qwen2.5-3B-Instruct / sharegpt** (A40, flashinfer)

| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |
|---|---|---|---|---|---|---|---|
| polyserve | `vllm/bf16/ctx8192/b512/gmu0.95/kvfp8_e5m2` | 1832 |  | 354 ms | 15.7 ms | 7876 ms | met |
| -kv | `vllm/bf16/ctx8192/b256/gmu0.95` | 1697 | -7.4% | 282 ms | 15.7 ms | 7161 ms | met |
| -kv (FlashInfer kept) | `vllm/bf16/ctx8192/b256/gmu0.95/attention_backend=FLASHINFER` | 1705 | -6.9% | 314 ms | 15.8 ms | 7279 ms | met |


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

## Task accuracy (GSM8K)

| model | weights | accuracy | 95% CI | vs reference | lost / gained | p (McNemar) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | bf16 | 86.4% | 84.4%–88.1% | reference | | |
| Qwen2.5-3B-Instruct | w8a8 | 85.1% | 83.0%–86.9% | -1.3 pts vs bf16 | 66 / 49 | 0.135 |
| Qwen2.5-3B-Instruct | bf16 | 87.2% | 85.3%–88.9% | reference | | |
| Qwen2.5-3B-Instruct | fp8 | 84.8% | 82.8%–86.7% | -2.4 pts vs bf16 | 65 / 34 | 0.002 |
| Qwen2.5-3B-Instruct | awq | 82.2% | 80.0%–84.2% | -5.0 pts vs bf16 | 119 / 53 | <0.001 |
| Qwen2.5-3B-Instruct | gptq | 83.1% | 81.0%–85.0% | -4.1 pts vs bf16 | 103 / 49 | <0.001 |
| Qwen2.5-7B-Instruct | bf16 | 91.4% | 89.7%–92.8% | reference | | |
| Qwen2.5-7B-Instruct | fp8 | 91.3% | 89.6%–92.7% | -0.1 pts vs bf16 | 31 / 30 | 1.000 |
| Qwen2.5-7B-Instruct | bf16 | 91.6% | 90.0%–93.0% | reference | | |
| Qwen2.5-7B-Instruct | fp8 | 91.4% | 89.7%–92.8% | -0.2 pts vs bf16 | 23 / 20 | 0.761 |
| Qwen2.5-7B-Instruct | awq | 90.9% | 89.2%–92.3% | -0.7 pts vs bf16 | 41 / 32 | 0.349 |
| Qwen2.5-7B-Instruct | gptq | 90.5% | 88.8%–92.0% | -1.1 pts vs bf16 | 46 / 32 | 0.141 |
