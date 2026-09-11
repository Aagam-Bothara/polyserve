# Benchmark matrix

Every cell is one `polyserve compare` run: PolyServe's calibrated pick, each installed backend's stock defaults, and real Ollama, all measured on the same workload with the same driver within minutes of each other. Raw results live in [results/](results/) as JSON; `polyserve report` turns them into [RESULTS.md](RESULTS.md) and the throughput-vs-TTFT plot.

## Matrix

| GPU | Model | vLLM | SGLang | llama.cpp | Ollama | PolyServe |
|---|---|---|---|---|---|---|
| RTX 3090 24 GB | Qwen2.5-3B-Instruct | ✓ | not installed | ✓ | ✓ | ✓ |
| A100 80 GB | Llama-3.2-3B-Instruct | | | | | |
| A100 80 GB | Llama-3.1-8B-Instruct | | | | | |
| RTX 3090 24 GB | Qwen2.5-7B-Instruct | | | | | |
| A30 | Llama-3.2-3B-Instruct | | | | | |
| GTX 1080 8 GB | Qwen2.5-1.5B-Instruct | n/a (cc 6.1) | n/a | | | |
| CPU only | Qwen2.5-1.5B-Instruct | n/a | n/a | | | |

The RTX 3090 row is complete for all six workloads: stock vLLM, stock llama.cpp and real Ollama were each measured against PolyServe's pick on the same card within minutes of each other. SGLang was not installed on that machine, so it has never been benchmarked. Every other row is empty.

Each row is run for every workload in `polyserve workloads` (`default`, `chat`, `long-context`, `generation`, `high-concurrency`, `rag`). A ✓ means the results file exists; `polyserve report` fills in the numbers.

## Running a machine

```bash
pip install "polyserve[nvml]" "vllm==0.11.0" "transformers>=4.56,<5"   # + sglang, llama.cpp, ollama as available
export LLAMA_SERVER=/path/to/llama-server
python benchmarks/run_matrix.py --models meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct
```

`run_matrix.py` runs `polyserve compare` for every (model, workload) pair that does not already have a results file, so it is safe to re-run after a crash. Pass `--workloads chat rag` to narrow it, `--ollama` to include Ollama rows (needs the `ollama` binary), `--objective latency` for a different objective.

Then, on any machine with the results directory:

```bash
polyserve report                   # writes benchmarks/RESULTS.md + benchmarks/throughput_vs_ttft.svg
```

## What each row means

- **PolyServe**: the profile `polyserve serve` would use (cached if present, calibrated otherwise), re-measured alongside the references. Its `calib` column is the one-time calibration cost.
- **`<backend>-default`**: the backend launched as its docs launch it, no flags beyond the model: `vllm serve <model>`, `sglang.launch_server --model-path`, `llama-server -m <Q4_K_M.gguf>`.
- **`ollama-default`**: the real `ollama serve` with `ollama pull <tag>`, Ollama's own quant and settings.
- **SLO**: whether the row's TTFT at its scored concurrency is under the workload's ceiling. `polyserve report` compares PolyServe against the *best default that meets the SLO*; defaults that miss it are shown but do not count as the baseline.
