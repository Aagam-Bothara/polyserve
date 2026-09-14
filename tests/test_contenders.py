"""Stages 3b-3f run for every engine within contender_band of the leader, not only the leader."""

from __future__ import annotations

from typing import Callable, List

from polyserve.calibrate.search import StagedSearch
from polyserve.models import Config, TrialMetrics, TrialResult

VLLM = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9)
SGLANG = Config(backend="sglang", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9)


class ScoredRunner:
    def __init__(self, score: Callable[[Config], float]):
        self.score, self.ran = score, []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.ran.append((stage, cfg))
        m = TrialMetrics(tok_s=self.score(cfg), ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048,
                         concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


def kv(c: Config) -> List[Config]:
    return [] if c.kv_dtype == "fp8" else [c.model_copy(update={"kv_dtype": "fp8"})]


def spec(c: Config) -> List[Config]:  # only vLLM offers speculative decoding here
    return [] if c.backend != "vllm" or c.spec_decode else [c.model_copy(update={"spec_decode": "ngram:4"})]


def dolly_like(sglang_base: float):
    """vLLM leads after the batch stage; SGLang gains far more from the fp8 cache than vLLM does."""
    def score(c: Config) -> float:
        tok = {"vllm": 500.0, "sglang": sglang_base}[c.backend]
        if c.kv_dtype == "fp8":
            tok *= {"vllm": 1.04, "sglang": 1.25}[c.backend]
        if c.spec_decode:
            tok *= 1.02
        return tok
    return score


def test_an_engine_close_behind_is_tuned_too_and_can_win():
    runner = ScoredRunner(dolly_like(480.0))
    search = StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)])
    winner, notes = search.run([VLLM, SGLANG])
    assert winner.config.backend == "sglang" and winner.config.kv_dtype == "fp8"  # 600 against vLLM's best
    assert any("sglang came within 10%" in n for n in notes)
    assert not any(c.backend == "sglang" and c.spec_decode for _, c in runner.ran)  # no vLLM change carried over


def test_the_leader_only_search_misses_it():
    search = StagedSearch(objective="throughput", runner=ScoredRunner(dolly_like(480.0)), contender_band=0,
                          variant_stages=[("kv", kv), ("spec", spec)])
    winner, _ = search.run([VLLM, SGLANG])
    assert winner.config.backend == "vllm"


def test_an_engine_far_behind_is_not_tuned():
    runner = ScoredRunner(dolly_like(400.0))  # 20% behind after the batch stage
    StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)]).run(
        [VLLM, SGLANG])
    assert [s for s, c in runner.ran if c.backend == "sglang"] == ["quant"]


def test_with_a_budget_the_variations_come_before_the_sweeps():
    clock = {"now": 0.0}

    class ClockRunner(ScoredRunner):
        def run(self, cfg, stage):
            clock["now"] += 60.0
            return super().run(cfg, stage)

    feasible = [VLLM.model_copy(update={"batch": b}) for b in (16, 64, 256)]
    runner = ClockRunner(lambda c: 500.0 * (1.3 if c.spec_decode else 1.0) * (1.01 if c.batch == 256 else 1.0))
    search = StagedSearch(objective="throughput", runner=runner, budget_s=180, clock=lambda: clock["now"],
                          variant_stages=[("kv", kv), ("spec", spec)])
    winner, _ = search.run(feasible)
    assert [s for s, _ in runner.ran] == ["quant", "kv", "spec"]  # three trials fit, and speculation is one
    assert winner.config.spec_decode == "ngram:4"


def test_with_a_budget_an_engine_behind_after_stage_one_gets_the_variations_too():
    """Dolly-15k on an A40: SGLang led the first trial (490 against 471), the gain was vLLM's."""
    clock = {"now": 0.0}

    class ClockRunner(ScoredRunner):
        def run(self, cfg, stage):
            clock["now"] += 60.0
            return super().run(cfg, stage)

    runner = ClockRunner(lambda c: {"vllm": 471.0, "sglang": 490.0}[c.backend] * (1.25 if c.spec_decode else 1.0))
    search = StagedSearch(objective="throughput", runner=runner, budget_s=300, clock=lambda: clock["now"],
                          variant_stages=[("kv", kv), ("spec", spec)])
    winner, _ = search.run([VLLM, SGLANG])
    assert winner.config.backend == "vllm" and winner.config.spec_decode == "ngram:4"
    assert [s for s, _ in runner.ran] == ["quant", "quant", "kv", "kv", "spec"]
