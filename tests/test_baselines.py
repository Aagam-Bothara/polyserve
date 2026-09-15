"""The baselines the staged search is measured against: a rule of thumb, and random search on the same time."""

from __future__ import annotations

import pytest

from polyserve.bench import compare
from polyserve.bench.baselines import random_search, rule_of_thumb, search_space
from polyserve.calibrate.objectives import Constraints
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config, TrialMetrics, TrialResult
from polyserve.pipeline import calibrate, combination_fits, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner


@pytest.fixture
def planned(hw_a100, spec, no_network):
    candidates, reg = select(hw_a100, spec, force="vllm")
    return prepare_and_plan(hw_a100, spec, candidates, reg), reg


def test_the_space_is_what_calibration_can_reach_each_once(hw_a100, planned):
    plan, reg = planned
    space = search_space(hw_a100, plan, reg, get_workload("chat"))
    keys = [c.key() for c in space]
    assert len(keys) == len(set(keys)) > len(plan.all_feasible)  # the variations multiply the planner's grid
    fits = combination_fits(hw_a100, reg, plan)
    assert all(fits(c) for c in space)
    assert any(c.spec_decode for c in space) and any(c.prefill_budget for c in space)


def test_the_rule_of_thumb_is_one_plausible_launch(hw_a100, planned):
    plan, reg = planned
    cfg = rule_of_thumb(hw_a100, plan, reg)
    assert cfg.backend == "vllm" and cfg.quant in ("fp8", "bf16")
    assert cfg.batch <= 256 and combination_fits(hw_a100, reg, plan)(cfg)


class TimedRunner:
    """Each trial takes 60 s on a fake clock; throughput grows with batch."""

    def __init__(self, clock):
        self.clock, self.ran = clock, []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.clock["now"] += 60.0
        self.ran.append(cfg.key())
        m = TrialMetrics(tok_s=float(cfg.batch), ttft_ms=100, tpot_ms=10, requests=16, output_tokens=2048, concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


SPACE = [Config(backend="vllm", quant="bf16", batch=b) for b in range(8, 408, 8)]  # 50 configurations


def _search(seed, budget=300):
    clock = {"now": 0.0}
    runner = TimedRunner(clock)
    run = random_search(SPACE, runner, "throughput", Constraints(), budget, seed=seed, clock=lambda: clock["now"])
    return run, runner


def test_random_search_keeps_to_its_budget_and_picks_the_best_it_measured():
    run, runner = _search(0)
    assert len(run.results) == 5 and run.seconds == 300 and run.untried == 45  # 5 trials of 60 s fit in 300 s
    assert run.winner.metrics.tok_s == max(r.metrics.tok_s for r in run.results)


def test_a_seed_fixes_the_order():
    assert _search(0)[1].ran == _search(0)[1].ran != _search(1)[1].ran


@pytest.mark.usefixtures("no_network")
def test_other_searches_picks_are_compared_as_rows_of_their_own(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg, runner=FakeRunner())
    rule = rule_of_thumb(hw_a100, plan, reg)
    result = compare(hw_a100, spec, profile, plan.prepared, reg, runner=CompareRunner(), include=["vllm-default"],
                     extra={"rule-of-thumb": rule})
    assert [r.label for r in result.rows] == ["polyserve", "vllm-default", "rule-of-thumb"]
