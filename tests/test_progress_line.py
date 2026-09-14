"""The calibration log reports a trial at the level the objective scores it, not at its fastest level."""

from __future__ import annotations

from polyserve.calibrate.objectives import Constraints
from polyserve.cli import _progress_line
from polyserve.models import Config, TrialMetrics, TrialResult

CFG = Config(backend="vllm", quant="fp8", batch=64)


def _trial(levels) -> TrialResult:
    """levels: {concurrency: (tok/s, TTFT p50, TTFT p95)}"""
    by = {str(c): TrialMetrics(tok_s=t, ttft_ms=p50, ttft_p95_ms=p95, tpot_ms=30, requests=16, output_tokens=2048,
                               concurrency=c) for c, (t, p50, p95) in levels.items()}
    best = max(by.values(), key=lambda m: m.tok_s)
    return TrialResult(config=CFG, stage="batch", metrics=best.model_copy(update={"by_concurrency": by}))


# The L4 on sharegpt: 32 users ran at 695 tok/s but broke the 1000 ms ceiling at p95; 8 users met it.
L4 = _trial({8: (219.0, 174.0, 420.0), 32: (695.0, 870.0, 1400.0)})


def test_the_scored_level_is_shown():
    line = _progress_line("confirm", CFG, L4, "balanced", Constraints(ttft_ceiling_ms=1000))
    assert "219.0 tok/s at 8 users" in line and "695" not in line and "breaks" not in line


def test_without_an_objective_it_shows_the_fastest_level():
    assert "695.0 tok/s at 32 users" in _progress_line("batch", CFG, L4)


def test_a_trial_that_breaks_the_limits_everywhere_says_so():
    slow = _trial({8: (219.0, 900.0, 1300.0)})
    assert "breaks the limits at every level" in _progress_line("batch", CFG, slow, "balanced",
                                                                 Constraints(ttft_ceiling_ms=1000))


def test_running_and_failed_trials():
    assert _progress_line("quant", CFG, None).endswith("...")
    failed = TrialResult(config=CFG, stage="quant", metrics=TrialMetrics(), launched=False, error="out of memory\nmore")
    assert "FAILED: out of memory" in _progress_line("quant", CFG, failed, "balanced", Constraints())
