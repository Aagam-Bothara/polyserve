# Contributing

## Setup

```bash
git clone https://github.com/Aagam-Bothara/polyserve.git && cd polyserve
pip install -e ".[dev]"
pytest                                  # no GPU or backend needed
ruff check polyserve tests benchmarks
```

CI runs both on Python 3.10–3.13 for every push and pull request.

## Tests

- Every test runs without a GPU, a backend or network access. Backends are driven through fake servers and fake hardware descriptors (see the fixtures in `tests/conftest.py`).
- Tests must not depend on the order a filesystem lists a directory in, or on anything else that varies by machine. Sort, and break ties explicitly.
- A bug found on real hardware gets a test that reproduces it with the numbers that were measured (`tests/test_combinations.py` is an example).

## Adding a backend

Implement the `Backend` interface in [docs/usage.md](docs/usage.md#backend-interface), register it with the selector, and give it a memory model the planner can use. The core pipeline (probe, plan, calibrate, serve) does not change.

## Benchmark claims

A number goes into [docs/benchmarks.md](docs/benchmarks.md) only if it was measured like this:

1. PolyServe's pick against stock settings on the same card, minutes apart: `polyserve compare <model> --workload W`. For any gain under about 10%, add `--repeats 3`: rows are measured interleaved, and a gain whose runs overlap the stock row's is flagged as within noise.
2. What each strategy was worth, flipped one at a time and measured back to back: `benchmarks/ablate_strategies.py`.
3. Any change of weights checked for quality: `benchmarks/task_quality.py` (GSM8K, paired against bf16).
4. The raw JSON committed under `benchmarks/strategies/`, and `SUMMARY.md` regenerated with `benchmarks/summarize_strategies.py` (the command is in its docstring).

State the GPU, engine versions and date. Report failures, losses and ties as well as wins; a strategy that lost is a result.

## Style

- `ruff`, line length 110.
- Comments say why, often with the measurement that motivated the code.
- Commit messages: one short imperative line.

## License

By contributing you agree that your contributions are licensed under the MIT license.
