# PolyServe

[![tests](https://github.com/Aagam-Bothara/polyserve/actions/workflows/tests.yml/badge.svg)](https://github.com/Aagam-Bothara/polyserve/actions/workflows/tests.yml)

**PolyServe is an autotuner for LLM serving: it finds the fastest configuration for your GPU and your traffic, then serves it.** Give it a model and, ideally, a sample of your prompts. It knows which settings each engine offers on each card (vLLM, SGLang and llama.cpp; weight precision, batch size, KV-cache type, speculative decoding and more), drops every configuration that will not fit in memory, measures the rest on your prompts, and serves the fastest one that meets your latency target behind an OpenAI-compatible API. What pays is what it tries and how it measures, not the order it tries things in. The winning settings changed from card to card and from one set of prompts to another, so PolyServe's pick beat a fixed rule of thumb (fp8 weights and KV cache, a short context, batch 256) by 15–97% on the three cards where both ran; but random sampling of the same settings, given the same time, did as well as PolyServe's staged order on two of those cards and better on the third. Calibration measures speed and latency; what quantization costs in answer quality is checked separately.

## Results

Measured on rented GPUs against stock `vllm serve` and SGLang defaults:

- **The pick holds on prompts it never saw.** Llama 3.1 8B, calibrated on 300 Dolly-15k prompts and measured on 300 others: **+93%** on an A40, **+59%** on an A100, **+71%** on an RTX 4090 over stock SGLang (stock vLLM does not start there), and **+24%** on an H100 over stock vLLM with fp8 weights.
- **Your traffic decides what pays.** On one A40 with Qwen2.5-3B the pick changed with the prompts: suffix decoding on Dolly prompts (**+122%** on held-out prompts), a draft model on news-article extraction (+66.5%), an fp8 KV cache on real chat (+10%).
- **Calibration pays for itself within hours.** It took 22–56 minutes per model and workload, repaid by 0.5–2.6 hours of busy serving on held-out prompts. Measuring the busiest load first skips 29–45% of that time without changing a pick.

Every result, with methods, ablations, quality checks and limits: [docs/benchmarks.md](docs/benchmarks.md).

## Quickstart

```bash
pip install git+https://github.com/Aagam-Bothara/polyserve.git   # not on PyPI yet
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

Linux, Python 3.10–3.13. NVIDIA GPUs of compute capability 7.5 or newer run vLLM, SGLang and llama.cpp (measured on an A40, A100, H100 NVL, L4 and RTX 4090); older NVIDIA GPUs and x86 CPUs run llama.cpp. vLLM-CPU and pre-Turing GPUs have never been benchmarked ([details](docs/benchmarks.md#hardware-and-engines-measured)).

## Learn more

- [docs/benchmarks.md](docs/benchmarks.md): every result, at a glance and in full, with methods, ablations, quality, limits and what is still unmeasured.
- [docs/usage.md](docs/usage.md): how calibration works, workloads, objectives, every search option and the CLI.
- [docs/writeup.md](docs/writeup.md): the design of the memory planner, calibration and the predictor, what the evidence does and does not support, and the roadmap.
- [benchmarks/strategies/SUMMARY.md](benchmarks/strategies/SUMMARY.md): every table, regenerated from the raw JSON.
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup, tests and adding a backend.

MIT licensed.
