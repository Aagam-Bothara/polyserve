"""Stage 5: the best few configurations are re-measured, and the choice is made on the fresh runs alone."""

from __future__ import annotations

from typing import Dict, List

from polyserve.calibrate.search import StagedSearch
from polyserve.cli import _opts
from polyserve.models import Config, TrialMetrics, TrialResult
from polyserve.pipeline import SearchOptions

FEASIBLE = [Config(backend="vllm", quant="bf16", ctx=8192, batch=b, gpu_memory_utilization=0.9) for b in (16, 64, 256)]


class SequenceRunner:
    """Each config's successive trials measure the next tok/s in its list (the last one repeats)."""

    def __init__(self, tok_s: Dict[int, List[float]], clock=None):
        self.tok_s, self.clock, self.ran = tok_s, clock, []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        n = sum(c.key() == cfg.key() for _, c in self.ran)
        self.ran.append((stage, cfg))
        if self.clock is not None:
            self.clock["now"] += 60.0
        seq = self.tok_s[cfg.batch]
        m = TrialMetrics(tok_s=seq[min(n, len(seq) - 1)], ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048,
                         concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


# batch 256 is lucky on its first trial (603, like the Dolly pick) and measures 540 after that.
LUCKY = {16: [480.0], 64: [560.0], 256: [603.0, 540.0]}


def test_without_it_a_lucky_trial_wins():
    winner, _ = StagedSearch(objective="throughput", runner=SequenceRunner(LUCKY)).run(FEASIBLE)
    assert winner.config.batch == 256 and winner.metrics.tok_s == 603.0


def test_the_choice_is_made_on_fresh_runs():
    runner = SequenceRunner(LUCKY)
    winner, notes = StagedSearch(objective="throughput", runner=runner, confirm_top=3).run(FEASIBLE)
    assert winner.config.batch == 64 and winner.stage == "confirm"
    # 480 is more than 10% behind 603, so only two configurations were worth re-measuring.
    assert [c.batch for s, c in runner.ran if s == "confirm"] == [256, 64]
    assert any("re-measured the best 2 configurations" in n and "560 tok/s in the search and 560" in n
               for n in notes)


def test_rounds_alternate_and_a_config_is_judged_by_its_worse_middle_run():
    runner = SequenceRunner({16: [480.0], 64: [560.0, 565.0, 550.0], 256: [603.0, 540.0, 530.0]})
    winner, _ = StagedSearch(objective="throughput", runner=runner, confirm_top=3, confirm_rounds=2).run(FEASIBLE)
    assert [c.batch for s, c in runner.ran if s == "confirm"] == [256, 64, 256, 64]
    assert winner.config.batch == 64 and winner.metrics.tok_s == 550.0


def test_a_spent_budget_skips_it():
    clock = {"now": 0.0}
    search = StagedSearch(objective="throughput", runner=SequenceRunner(LUCKY, clock), clock=lambda: clock["now"])
    search.run(FEASIBLE)
    search.budget_s, search.confirm_top = 1.0, 3
    assert search.stage_confirm() is None
    assert any(s.startswith("confirm ") for s in search._skipped)


def test_it_is_on_by_default_and_cached_apart_when_off():
    assert SearchOptions().confirm and _opts("auto", "on", "on", "on").confirm
    assert not _opts("auto", "on", "on", "on", confirm="off").confirm
    assert SearchOptions(confirm=False).key() == {"confirm": "off"}
