# PolyServe

[![tests](https://github.com/Aagam-Bothara/polyserve/actions/workflows/tests.yml/badge.svg)](https://github.com/Aagam-Bothara/polyserve/actions/workflows/tests.yml)

**PolyServe finds the fastest way to serve an LLM on your GPU and your traffic, then serves it behind an OpenAI-compatible API.** It measures real configurations — engine, weight precision, batch size, KV-cache type, speculative decoding — instead of guessing, because the settings that won changed on every card tried.

**It beats a hand-tuned configuration by 56–97%**, which is the comparison that matters — not the one against untouched defaults.

| Llama 3.1 8B, held-out prompts | PolyServe | a fixed rule of thumb † | (stock defaults) |
|---|---|---|---|
| RTX 4090 | **1041 tok/s** | 667 — **+56%** | SGLang 448 |
| A100 | **1155 tok/s** | 674 — **+71%** | vLLM 726 |
| A40 | **506 tok/s** | 257 — **+97%** | vLLM 262 |

† fp8 weights and KV cache, a short context, batch 256: what an informed engineer sets without measuring. That rule was **slower than changing nothing** on both Ampere cards, because an fp8 KV cache costs throughput there — which is the point: the settings that win move from card to card, so a rule that is right somewhere is wrong elsewhere. The stock column is context rather than a claim; nobody serious ships untouched defaults. Each row was measured on prompts held out of calibration ([full sessions](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md#llama-31-8b-on-four-gpus)).

## Results

- **Your traffic decides what pays, not the card alone.** On one A40 with Qwen2.5-3B the winning setting changed with the prompts: suffix decoding on Dolly prompts (+122% over stock), a draft model on news-article extraction (+66.5%), an fp8 KV cache on real chat (+10%). A tuner that ignores your prompts cannot find these.
- **The pick holds on prompts it never saw.** Every number here was measured on prompts held out of calibration: 300 Dolly-15k prompts to tune on, 300 others to score on. An H100 not in the table gained +24% over stock vLLM with fp8 weights.
- **Calibration pays for itself within hours.** 22–56 minutes per model and workload, repaid by 0.5–2.6 hours of busy serving. Measuring the busiest load first cuts 29–45% off that without changing a pick.
- **What the search order is worth: one win, one tie.** Given the same time, random sampling of the same space matched this staged search on all three 8B cards. On a 14B in fp8 on a 24 GB card the staged search won by 4.4% (385 against 369 tok/s); on the same card in the 7× larger space that includes 4-bit checkpoints, the two tied (1412 against 1416) even though random search was given less time ([the evidence](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md#qwen25-14b-where-memory-binds)). What staging reliably gives is the same answer every run, every strategy tried at least once, and a profile saying what each setting was worth — the gains come from measuring, not from the order.

Every result, with methods, ablations, quality checks and limits: [docs/benchmarks.md](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/benchmarks.md).

## How this differs from vLLM's auto-tune and AIConfigurator

Not benchmarked against either — the comparison below is one of *kind*, not of measured results, and saying otherwise would be the overclaiming this project tries to avoid.

- **vLLM's `auto_tune`** sweeps one engine's throughput knobs — batch size and batched-token budget — to find the best setting that still meets a latency target. Same spirit, narrower question: it tunes vLLM once you have already chosen vLLM, your weight precision and your decoding strategy. PolyServe treats the engine itself as a variable, alongside precision, KV-cache type and speculative decoding — the dimensions that carried almost all the gain here. On an RTX 4090 SGLang led the early trials and vLLM only overtook it once speculative decoding and a quantized cache were tried — tuning the early leader alone would have eliminated the eventual winner, and a single-engine sweep never sees the crossover at all.
- **AIConfigurator** predicts good deployments analytically from a performance model, aimed at TensorRT-LLM, and answers sizing questions — how many GPUs, which parallelism — very cheaply because it does not launch anything. PolyServe measures instead, which costs 20–50 minutes per workload and is why its answers are specific to your prompts. We tried the analytical shortcut and kept it small for a reason: our own fitted predictor reaches a rank correlation of 0.88–0.94, enough to prune quantizations that cannot win, but it ranks the true winner at a median of 15th out of 25 measured configurations — so prediction is a filter here, never the decision ([why](https://github.com/Aagam-Bothara/polyserve/blob/main/docs/decisions.md)).
- **What none of the three does** is prove the others wrong. The honest gap is that no one outside this project has run PolyServe, and we have not run their tools on our hardware.

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
