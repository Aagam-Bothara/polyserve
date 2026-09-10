from __future__ import annotations

import math
from typing import List

import pytest

from polyserve.backends import get_backend
from polyserve.calibrate.objectives import Constraints, pick, rank
from polyserve.calibrate.search import StagedSearch, _baseline
from polyserve.memory import plan
from polyserve.models import Config, TrialMetrics, TrialResult


def _res(key: str, tok_s: float, ttft, jpt=None, ok=True, quant="bf16") -> TrialResult:
    cfg = Config(backend="vllm", quant=quant, ctx=4096, batch=int(key), gpu_memory_utilization=0.9)
    m = TrialMetrics(tok_s=tok_s, ttft_ms=ttft, tpot_ms=10, joules_per_token=jpt, requests=16,
                     failed=0 if ok else 16, output_tokens=2048 if ok else 0)
    return TrialResult(config=cfg, stage="t", metrics=m, error=None if ok else "boom")


def _relaxed(notes):
    return any(("least-violating" in n) or ("no energy" in n) for n in notes)


RESULTS = [
    _res("16", tok_s=800, ttft=120, jpt=0.30),
    _res("64", tok_s=1500, ttft=400, jpt=0.20),
    _res("256", tok_s=1900, ttft=900, jpt=0.18),
    _res("8", tok_s=0, ttft=None, ok=False),
]


def test_throughput_is_plain_argmax():
    w, notes = pick(RESULTS, "throughput")
    assert w.config.batch == 256 and not _relaxed(notes)


def test_balanced_respects_ttft_ceiling():
    w, notes = pick(RESULTS, "balanced", Constraints(ttft_ceiling_ms=500))
    assert w.config.batch == 64 and not _relaxed(notes)


def test_balanced_relaxes_when_nothing_fits():
    w, notes = pick(RESULTS, "balanced", Constraints(ttft_ceiling_ms=50))
    assert w.config.batch == 16  # least violation
    assert any("least-violating" in n for n in notes)


def test_latency_min_ttft_subject_to_floor():
    # floor = 0.5 * 1900 = 950 -> batch 16 (800 tok/s) is below the floor.
    w, _ = pick(RESULTS, "latency", Constraints(tok_s_floor_frac=0.5))
    assert w.config.batch == 64
    w, _ = pick(RESULTS, "latency", Constraints(tok_s_floor_abs=100))
    assert w.config.batch == 16


def test_efficiency_min_joules_subject_to_floor():
    w, notes = pick(RESULTS, "efficiency", Constraints(tok_s_floor_abs=1000))
    assert w.config.batch == 256 and not _relaxed(notes)


def test_efficiency_without_energy_falls_back_to_tok_s():
    rs = [_res("16", 800, 100), _res("64", 1500, 200)]
    w, notes = pick(rs, "efficiency", Constraints(tok_s_floor_abs=0))
    assert w.config.batch == 64 and any("no energy" in n for n in notes)


def test_rank_puts_failed_trials_nowhere_and_infeasible_last():
    ranked = rank(RESULTS, "balanced", Constraints(ttft_ceiling_ms=500))
    assert [r.result.config.batch for r in ranked] == [64, 16, 256]
    assert [r.feasible for r in ranked] == [True, True, False]


def test_unknown_objective():
    with pytest.raises(ValueError):
        pick(RESULTS, "vibes")


# --------------------------------------------------------------------------- staged search


class FakeRunner:
    """Deterministic synthetic performance model; gmu 0.95 configs fail to launch (simulated OOM)."""

    QUANT_SPEED = {"bf16": 1.0, "fp8": 1.35, "Q4_K_M": 0.9, "Q5_K_M": 0.8, "Q6_K": 0.7, "Q8_0": 0.6}

    def __init__(self):
        self.calls: List[str] = []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.calls.append(cfg.key())
        if cfg.gpu_memory_utilization is not None and cfg.gpu_memory_utilization >= 0.95:
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error="CUDA OOM")
        speed = self.QUANT_SPEED[cfg.quant] * (1 + 0.5 * math.log2(cfg.batch)) * 100
        ttft = 60 + 3 * cfg.batch + cfg.ctx / 100
        m = TrialMetrics(tok_s=speed, ttft_ms=ttft, tpot_ms=1000 / speed, joules_per_token=1 / speed,
                         power_w=250, peak_mem_mb=10_000, requests=48, failed=0, output_tokens=48 * 128)
        return TrialResult(config=cfg, stage=stage, metrics=m)


def _feasible_a100(hw_a100, prepared_vllm) -> List[Config]:
    be = get_backend("vllm")
    return [c for c, _ in plan(hw_a100, prepared_vllm, be.candidate_configs(hw_a100, prepared_vllm), be.memory_model(hw_a100))]


def test_baseline_is_median_ctx_batch_with_largest_memory():
    cfgs = [Config(backend="vllm", quant="bf16", ctx=c, batch=b, gpu_memory_utilization=g)
            for c in (2048, 4096, 8192) for b in (16, 64, 256) for g in (0.8, 0.9, 0.95)]
    b = _baseline(cfgs)
    assert (b.ctx, b.batch, b.gpu_memory_utilization) == (4096, 64, 0.95)
    lc = [Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=n) for n in (18, 27, 37)]
    assert _baseline(lc).n_gpu_layers == 37  # full offload, not the median


def test_objective_scores_each_trial_at_its_best_level():
    def res(batch, levels):
        cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=batch, gpu_memory_utilization=0.9)
        by = {str(c): TrialMetrics(tok_s=t, ttft_ms=ttft, concurrency=c, requests=16, output_tokens=100)
              for c, t, ttft in levels}
        best = max(by.values(), key=lambda m: m.tok_s)
        return TrialResult(config=cfg, stage="t", metrics=TrialMetrics(
            tok_s=best.tok_s, ttft_ms=best.ttft_ms, requests=48, output_tokens=300, by_concurrency=by))
    # Config A: fastest at c=8 but TTFT blows the ceiling there; fine at c=4.
    a = res(16, [(1, 200, 40), (4, 600, 120), (8, 900, 2000)])
    # Config B: slower everywhere but always under the ceiling.
    b = res(64, [(1, 180, 40), (4, 500, 100), (8, 650, 300)])
    from polyserve.calibrate.objectives import rank
    ranked = rank([a, b], "balanced", Constraints(ttft_ceiling_ms=500, noise_tolerance=0))
    assert ranked[0].result is b and ranked[0].concurrency == 8  # 650 @ c8 beats A's 600 @ c4
    assert ranked[1].concurrency == 4  # A scored at its best feasible level, not its fastest
    w, notes = pick([a, b], "throughput")
    assert w is a and any("concurrency 8" in n for n in notes)


def test_noise_tolerance_prefers_larger_context():
    small = _res("64", tok_s=1064.7, ttft=31)
    big = _res("64", tok_s=1053.3, ttft=32)
    big.config.ctx = 8192
    w, _ = pick([small, big], "throughput", Constraints(noise_tolerance=0.02))
    assert w is big
    w, _ = pick([small, big], "throughput", Constraints(noise_tolerance=0.0))
    assert w is small


def test_staged_search_runs_stages_and_dedups(hw_a100, prepared_vllm):
    feasible = _feasible_a100(hw_a100, prepared_vllm)
    runner = FakeRunner()
    search = StagedSearch(objective="throughput", runner=runner)
    winner, notes = search.run(feasible)
    assert winner is not None
    stages = [r.stage for r in search.results]
    # Stage 1 tries gmu 0.95 first (fails: simulated OOM) then 0.90 for each of bf16 and fp8.
    assert stages.count("quant") == 4
    assert sum(1 for r in search.results if r.stage == "quant" and r.ok) == 2
    assert "memory" in stages and "batch" in stages
    assert len(runner.calls) == len(set(runner.calls))  # no config run twice
    assert len(runner.calls) < len(feasible) / 3  # staged, not a grid
    # throughput: fp8 is faster in the fake model, batch 256 the best batch.
    assert winner.config.quant == "fp8" and winner.config.batch == 256
    # stage 2 must have skipped the OOM-ing 0.95 and settled on 0.90 ("largest safe").
    assert winner.config.gpu_memory_utilization == 0.90
    assert any(r.stage == "memory" and r.error == "CUDA OOM" for r in search.results)


def test_staged_search_balanced_uses_ceiling(hw_a100, prepared_vllm):
    feasible = _feasible_a100(hw_a100, prepared_vllm)
    search = StagedSearch(objective="balanced", runner=FakeRunner(), constraints=Constraints(ttft_ceiling_ms=300))
    winner, notes = search.run(feasible)
    assert winner.metrics.ttft_ms <= 300
    assert winner.config.batch == 64 and not _relaxed(notes)


def test_staged_search_keeps_top_two_quants_only(hw_gtx1080, prepared_llamacpp):
    be = get_backend("llamacpp-cuda")
    grid = be.candidate_configs(hw_gtx1080, prepared_llamacpp)
    feasible = [c for c, _ in plan(hw_gtx1080, prepared_llamacpp, grid, be.memory_model(hw_gtx1080))]
    runner = FakeRunner()
    search = StagedSearch(objective="throughput", runner=runner)
    winner, _ = search.run(feasible)
    quant_stage = {r.config.quant for r in search.results if r.stage == "quant"}
    later = {r.config.quant for r in search.results if r.stage != "quant"}
    assert len(quant_stage) >= 3 and len(later) == 2
    assert winner.config.quant == "Q4_K_M"


def test_staged_search_all_fail():
    class Dead:
        def run(self, cfg, stage):
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error="nope")

    cfgs = [Config(backend="vllm", quant="bf16", ctx=2048, batch=16, gpu_memory_utilization=0.8)]
    winner, notes = StagedSearch(objective="throughput", runner=Dead()).run(cfgs)
    assert winner is None and "every stage-1 trial failed" in notes[0]


def test_staged_search_empty():
    winner, notes = StagedSearch(objective="throughput", runner=FakeRunner()).run([])
    assert winner is None and "no feasible" in notes[0]
