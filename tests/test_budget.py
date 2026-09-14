"""--budget: calibration stops starting trials once the next would end past the budget."""

from __future__ import annotations

import pytest
import typer

from polyserve.calibrate.search import StagedSearch
from polyserve.cli import _duration
from polyserve.models import Config, TrialMetrics, TrialResult
from polyserve.pipeline import SearchOptions

FEASIBLE = [Config(backend="llamacpp-cuda", quant=q, ctx=4096, batch=b, n_gpu_layers=37, n_batch=512)
            for q in ("Q4_K_M", "Q8_0") for b in (1, 4, 8)]


class ClockRunner:
    """Every trial takes 60 s on a fake clock; throughput grows with the batch."""

    def __init__(self):
        self.now, self.ran = 0.0, []

    def clock(self) -> float:
        return self.now

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.now += 60.0
        self.ran.append(cfg)
        m = TrialMetrics(tok_s=100.0 + cfg.batch, ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048,
                         concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


def _search(budget):
    runner = ClockRunner()
    return runner, StagedSearch(objective="throughput", runner=runner, budget_s=budget, clock=runner.clock)


def test_the_budget_stops_new_trials_and_the_best_so_far_wins():
    runner, search = _search(150)
    winner, notes = search.run(FEASIBLE)
    assert len(runner.ran) == 2  # 60 s each: a third would end at 180 s
    assert winner is not None and winner.config in runner.ran
    assert all(r.launched for r in search.results)  # skipped trials are not results
    assert any("budget of 150s reached after 2 trials" in n for n in notes)


def test_one_trial_always_runs_and_no_budget_runs_everything():
    runner, search = _search(1)
    winner, _ = search.run(FEASIBLE)
    assert len(runner.ran) == 1 and winner is not None
    runner, search = _search(None)
    _, notes = search.run(FEASIBLE)
    assert len(runner.ran) > 2 and not any("budget" in n for n in notes)


def test_a_budgeted_profile_is_cached_under_its_own_name():
    assert SearchOptions(budget_s=600).key() == {"budget": "600s"} and SearchOptions().key() == {}


def test_duration_parsing():
    assert [_duration(v) for v in ("90s", "10m", "1h", "45", None)] == [90, 600, 3600, 45, None]
    for bad in ("abc", "0", "10d"):
        with pytest.raises(typer.BadParameter):
            _duration(bad)
