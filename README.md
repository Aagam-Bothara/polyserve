# PolyServe

[![tests](https://github.com/Aagam-Bothara/polyserve/actions/workflows/tests.yml/badge.svg)](https://github.com/Aagam-Bothara/polyserve/actions/workflows/tests.yml)

**PolyServe finds the fastest way to serve an LLM on your GPU and your traffic, then serves it behind an OpenAI-compatible API.** It measures real configurations — engine, weight precision, batch size, KV-cache type, speculative decoding — instead of guessing, because the settings that won changed on every card tried.

| Llama 3.1 8B, held-out prompts | PolyServe | an expert's rule of thumb † | stock default |
|---|---|---|---|
| RTX 4090 | **1041 tok/s** | 667 (+56%) | SGLang 448 (+132%) |
| A100 | **1155 tok/s** | 674 (+71%) | vLLM 726 (+59%) |
| A40 | **506 tok/s** | 257 (+97%) | vLLM 262 (+93%) |

† fp8 weights and KV cache, a short context, batch 256: what an informed user sets without measuring. On both Ampere cards it was **slower than changing nothing**, because an fp8 KV cache costs throughput there. Percentages are PolyServe's lead; each row was measured on prompts held out of calibration, in the sessions detailed in [benchmarks](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md#llama-31-8b-on-four-gpus).

## Results

- **Your traffic decides what pays, not the card alone.** On one A40 with Qwen2.5-3B the winning setting changed with the prompts: suffix decoding on Dolly prompts (+122% over stock), a draft model on news-article extraction (+66.5%), an fp8 KV cache on real chat (+10%). A tuner that ignores your prompts cannot find these.
- **The pick holds on prompts it never saw.** Every number here was measured on prompts held out of calibration: 300 Dolly-15k prompts to tune on, 300 others to score on. An H100 not in the table gained +24% over stock vLLM with fp8 weights.
- **Calibration pays for itself within hours.** 22–56 minutes per model and workload, repaid by 0.5–2.6 hours of busy serving. Measuring the busiest load first cuts 29–45% off that without changing a pick.
- **What the search order is worth: 4% where memory binds.** Given the same time, random sampling of the same space matched this staged search on all three 8B cards. On a 14B in fp8 on a 24 GB card — where only 2 of 54 configurations fit and stock vLLM cannot start at all — the staged search found a configuration random sampling missed: **385 against 369 tok/s** at equal time ([the evidence](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md#qwen25-14b-where-memory-binds)). Staging also gives the same answer every run, tries every strategy at least once, and leaves a profile saying what each setting was worth.

Every result, with methods, ablations, quality checks and limits: [docs/benchmarks.md](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md).

## Quickstart

```bash
pip install polyserve
pip install "vllm==0.29.0"     # driver 580+; older drivers: "vllm==0.11.0" "transformers>=4.56,<5"
polyserve serve Qwen/Qwen2.5-3B-Instruct --workload chat
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

The first launch calibrates and caches the result per machine, model, objective and workload; later launches serve at once. To tune for your own traffic, pass a sample of it: `--workload-file prompts.jsonl`, one prompt per line. Optional: `pip install arctic-inference==0.1.1` adds suffix decoding; SGLang can live in its own environment (set `SGLANG_PYTHON` to its `python`); llama.cpp needs `llama-server` built with CUDA, on `PATH` or in `LLAMA_SERVER`.

With Docker (built on the official vLLM image):

```bash
docker build -t polyserve .
docker run --gpus all --ipc=host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.polyserve:/root/.polyserve polyserve serve Qwen/Qwen2.5-3B-Instruct
```

### Supported hardware

Linux, Python 3.10–3.13. NVIDIA GPUs of compute capability 7.5 or newer run vLLM, SGLang and llama.cpp (measured on an A40, A100, H100 NVL, L4 and RTX 4090); older NVIDIA GPUs and x86 CPUs run llama.cpp. vLLM-CPU and pre-Turing GPUs have never been benchmarked ([details](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md#hardware-and-engines-measured)).

## Learn more

- [docs/benchmarks.md](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md): every result, at a glance and in full, with methods, ablations, quality, limits and what is still unmeasured.
- [docs/usage.md](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/usage.md): how calibration works, workloads, objectives, every search option and the CLI.
- [docs/writeup.md](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/writeup.md): the design of the memory planner, calibration and the predictor, what the evidence does and does not support, and the roadmap.
- [docs/decisions.md](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/decisions.md): why each choice is the way it is — including why not just random search — with the measurement that settled it.
- [benchmarks/strategies/SUMMARY.md](https://github.com/Aagam-Bothara/polyserve/blob/main/benchmarks/strategies/SUMMARY.md): every table, regenerated from the raw JSON.
- [CONTRIBUTING.md](https://github.com/Aagam-Bothara/polyserve/blob/main/CONTRIBUTING.md): development setup, tests and adding a backend.

MIT licensed.
