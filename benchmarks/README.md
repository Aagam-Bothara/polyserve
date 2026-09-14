# Benchmark matrix

> Current results, with methods and caveats, are in [docs/benchmarks.md](../docs/benchmarks.md); their raw files are in [strategies/](strategies/). This page describes the matrix runner (`run_matrix.py` and `polyserve report`). The RTX 3090 row was measured with an early harness that sent the same prompts at every concurrency level, which let prefix caching inflate throughput. Its numbers were withdrawn; its files stay in [results/](results/) only for the memory-planner figures, which caching does not affect.

Every cell is one or more `polyserve compare` runs: PolyServe's calibrated pick against each installed backend's stock defaults (and real Ollama when installed), measured on the same workload with the same driver within minutes of each other.

## Matrix

| GPU | Model | vLLM | SGLang | llama.cpp | Ollama | PolyServe |
|---|---|---|---|---|---|---|
| A40 48 GB | Qwen2.5-3B-Instruct | ✓ | ✓ (one workload) | ✓ | | ✓ |
| A40 48 GB | Qwen2.5-7B-Instruct | ✓ (`chat`) | | ✓ (`chat`) | | ✓ |
| L4 24 GB | Qwen2.5-7B-Instruct | ✓ | | ✓ | | ✓ |
| 2× A40, PCIe | Qwen2.5-3B-Instruct | ✓ (layouts, disaggregated) | | | | ✓ |
| CPU container, 7.65 cores | Qwen2.5-0.5B-Instruct | n/a | n/a | ✓ | | ✓ |
| RTX 3090 24 GB | Qwen2.5-3B-Instruct | withdrawn | not installed | withdrawn | withdrawn | withdrawn |
| A100 80 GB | Llama-3.2-3B / Llama-3.1-8B | | | | | |
| A30 | Llama-3.2-3B-Instruct | | | | | |
| GTX 1080 8 GB | Qwen2.5-1.5B-Instruct | n/a (cc 6.1) | n/a | | | |

A ✓ means at least one comparison exists for that pair; which workloads each covers is listed in [docs/benchmarks.md](../docs/benchmarks.md).

## Running a machine

```bash
pip install "polyserve[nvml]" "vllm==0.29.0"   # driver 580+; older drivers: "vllm==0.11.0" "transformers>=4.56,<5"
export LLAMA_SERVER=/path/to/llama-server       # optional
export SGLANG_PYTHON=/path/to/sglang-env/bin/python   # optional: SGLang in its own environment
python benchmarks/run_matrix.py --models meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct
```

`run_matrix.py` runs `polyserve compare` for every (model, workload) pair that does not already have a results file, so it is safe to re-run after a crash. Pass `--workloads chat rag` to narrow it, `--ollama` to include Ollama rows (needs the `ollama` binary), `--objective latency` for a different objective. For claims under about 10%, run `polyserve compare --repeats 3` instead, which measures each row three times interleaved and flags differences within noise.

Then, on any machine with the results directory:

```bash
polyserve report                   # writes benchmarks/RESULTS.md + benchmarks/throughput_vs_ttft.svg
```

## What each row means

- **PolyServe**: the profile `polyserve serve` would use (cached if present, calibrated otherwise), re-measured alongside the references. Its `calib` column is the one-time calibration cost.
- **`<backend>-default`**: the backend launched as its docs launch it, no flags beyond the model: `vllm serve <model>`, `sglang.launch_server --model-path`, `llama-server -m <Q4_K_M.gguf>`.
- **`ollama-default`**: the real `ollama serve` with `ollama pull <tag>`, Ollama's own quant and settings.
- **SLO**: whether the row's TTFT at its scored concurrency is under the workload's ceiling. `polyserve report` compares PolyServe against the *best default that meets the SLO*; defaults that miss it are shown but do not count as the baseline.
