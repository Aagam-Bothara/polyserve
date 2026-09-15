"""Stage 3g: random configurations of the leading engine and precision, for settings that only pay together."""

from __future__ import annotations

from typing import List

import typer

from polyserve.calibrate.search import StagedSearch
from polyserve.cli import app
from polyserve.models import Config
from polyserve.pipeline import SearchOptions, calibrate, prepare_and_plan, select
from tests.test_contenders import SGLANG, VLLM, ScoredRunner
from tests.test_objectives_and_search import FakeRunner

DRAFT = "draft:small:4"


def kv(c: Config) -> List[Config]:
    return [] if c.kv_dtype != "auto" else [c.model_copy(update={"kv_dtype": "int8"})]


def spec(c: Config) -> List[Config]:
    return [] if c.spec_decode else [c.model_copy(update={"spec_decode": DRAFT})]


def rtx4090_like(c: Config) -> float:
    """The draft model pays on its own; the int8 cache costs 10% alone and adds 25% with the draft model."""
    tok = 670.0 * (1.3 if c.spec_decode else 1.0)
    if c.kv_dtype == "int8":
        tok *= 1.25 if c.spec_decode else 0.9
    return tok


SPACE = [VLLM.model_copy(update={"kv_dtype": k, "spec_decode": s}) for k in ("auto", "int8") for s in (None, DRAFT)]


def test_exploring_finds_what_one_change_at_a_time_cannot():
    runner = ScoredRunner(rtx4090_like)
    search = StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)],
                          explore_space=SPACE)
    winner, notes = search.run([VLLM])
    assert (winner.config.kv_dtype, winner.config.spec_decode) == ("int8", DRAFT)
    assert [s for s, c in runner.ran if s == "explore"] == ["explore"]  # the one configuration the stages skipped
    assert any("explored 1 random configurations of vllm bf16" in n and "took the lead" in n for n in notes)


def test_without_an_explore_space_the_stages_keep_their_pick():
    runner = ScoredRunner(rtx4090_like)
    winner, _ = StagedSearch(objective="throughput", runner=runner,
                             variant_stages=[("kv", kv), ("spec", spec)]).run([VLLM])
    assert (winner.config.kv_dtype, winner.config.spec_decode) == ("auto", DRAFT)
    assert not any(s == "explore" for s, _ in runner.ran)


def test_it_draws_only_the_leaders_engine_and_precision():
    others = [SGLANG, VLLM.model_copy(update={"quant": "fp8", "batch": 16}), SGLANG.model_copy(update={"batch": 16})]
    runner = ScoredRunner(lambda c: rtx4090_like(c) if c.backend == "vllm" and c.quant == "bf16" else 100.0)
    StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)],
                 explore_space=SPACE + others, explore_min=10).run([VLLM, SGLANG])
    explored = [c for s, c in runner.ran if s == "explore"]
    assert explored and all(c.backend == "vllm" and c.quant == "bf16" for c in explored)


def test_calibration_explores_by_default_and_the_option_turns_it_off(hw_a100, spec, no_network):
    candidates, reg = select(hw_a100, spec, force="vllm")
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    on = calibrate(hw_a100, spec, "throughput", plan, reg, runner=FakeRunner())
    assert any(t.stage == "explore" for t in on.calibration_table)
    off = calibrate(hw_a100, spec, "throughput", plan, reg, runner=FakeRunner(), options=SearchOptions(explore=False))
    assert not any(t.stage == "explore" for t in off.calibration_table)
    assert SearchOptions(explore=False).key()["explore"] == "off" and "explore" not in SearchOptions().key()


def test_the_calibrating_commands_take_explore():
    commands = typer.main.get_command(app).commands
    for name in ("bench", "recalibrate", "compare", "serve"):
        assert "explore" in [p.name for p in commands[name].params]
