"""Speculation first, the other stages per speculation setting, engines that cannot catch up skipped, and a known
start-up failure left out of the search."""

from __future__ import annotations

from typing import List

from polyserve.calibrate.search import StagedSearch
from polyserve.models import Config
from polyserve.pipeline import calibrate, combination_fits, prepare_and_plan, select
from tests.test_contenders import SGLANG, ScoredRunner
from tests.test_objectives_and_search import FakeRunner

VLLM = Config(backend="vllm", quant="fp8", ctx=16384, batch=16, gpu_memory_utilization=0.95)
SGL = SGLANG.model_copy(update={"quant": "fp8", "batch": 16})
SPECS = {"suffix:24": 1.35, "draft:small:4": 1.15, "ngram_gpu:4": 1.15}


def rtx4090_like(c: Config) -> float:
    """SGLang in fp8 leads at first and has no speculation. On vLLM, suffix decoding is the best method alone,
    and the int8 cache and a 16k prefill budget pay only with the draft model (neutral or worse without it)."""
    if c.backend == "sglang":
        return 781.0 * (0.96 if c.kv_dtype != "auto" else 1.0)
    tok = 670.0 * SPECS.get(c.spec_decode or "", 1.0)
    draft = (c.spec_decode or "").startswith("draft")
    if c.kv_dtype == "int8":
        tok *= 1.2 if draft else 1.0
    if c.prefill_budget == 16384:
        tok *= 1.12 if draft else 0.99
    return tok


def kv(c: Config) -> List[Config]:
    if c.kv_dtype != "auto":
        return []
    return [c.model_copy(update={"kv_dtype": "int8" if c.backend == "vllm" else "fp8_e5m2"})]


def spec(c: Config) -> List[Config]:
    if c.backend != "vllm" or c.spec_decode:
        return []
    return [c.model_copy(update={"spec_decode": s}) for s in SPECS]


def prefill(c: Config) -> List[Config]:
    return [c.model_copy(update={"prefill_budget": b}) for b in (2048, 16384) if c.prefill_budget != b]


def search(spec_first: bool) -> tuple:
    runner = ScoredRunner(rtx4090_like)
    s = StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)],
                     prefill_variants=prefill, spec_first=spec_first)
    winner, notes = s.run([VLLM, SGL])
    return winner, notes, runner


def test_speculation_first_finds_what_only_pays_with_the_draft_model():
    winner, notes, runner = search(spec_first=True)
    assert winner.config.spec_decode == "draft:small:4"
    assert (winner.config.kv_dtype, winner.config.prefill_budget) == ("int8", 16384)
    stages = [s for s, c in runner.ran if c.backend == "vllm"]
    assert stages.index("spec") < stages.index("prefill")  # speculation before the settings that depend on it


def test_an_engine_left_behind_with_nothing_more_to_offer_is_not_tuned():
    _, notes, runner = search(spec_first=True)
    assert [s for s, c in runner.ran if c.backend == "sglang"] == ["quant"]  # only its first trial
    assert any("sglang was not tuned further: vllm led it by" in n for n in notes)


def test_sglangs_large_prefill_budgets_are_left_out_on_a_24_gb_card(hw_a100, spec, no_network):
    candidates, reg = select(hw_a100, spec)
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    small = hw_a100.model_copy(update={"gpus": [g.model_copy(update={"vram_total_bytes": 24 * 2**30})
                                                for g in hw_a100.gpus]})
    assert small.gpu.vram_total_bytes == 24 * 2**30
    sgl = plan.feasible["sglang"][0][0].model_copy(update={"prefill_budget": 8192})
    vll = plan.feasible["vllm"][0][0].model_copy(update={"prefill_budget": 8192})
    assert combination_fits(hw_a100, reg, plan)(sgl)
    assert not combination_fits(small, reg, plan)(sgl)
    assert combination_fits(small, reg, plan)(vll)  # the rule is SGLang's alone


def test_calibration_runs_speculation_before_the_prefill_stage(hw_a100, spec, no_network):
    candidates, reg = select(hw_a100, spec, force="vllm")
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    stages = [t.stage for t in calibrate(hw_a100, spec, "throughput", plan, reg, runner=FakeRunner()).calibration_table]
    assert "spec" in stages and "prefill" in stages and stages.index("spec") < stages.index("prefill")
