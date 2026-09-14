"""balanced judges the TTFT ceiling at the 95th percentile by default; the median is opt-in."""

from __future__ import annotations

import pytest
import typer

from polyserve.calibrate.objectives import Constraints, pick
from polyserve.cli import _percentile
from polyserve.models import Config, TrialMetrics, TrialResult
from polyserve.pipeline import SearchOptions


def _trial(batch: int, levels) -> TrialResult:
    """levels: {concurrency: (tok/s, TTFT p50, TTFT p95)}"""
    by = {str(c): TrialMetrics(tok_s=t, ttft_ms=p50, ttft_p95_ms=p95, tpot_ms=20, requests=16, output_tokens=2048,
                               concurrency=c) for c, (t, p50, p95) in levels.items()}
    best = max(by.values(), key=lambda m: m.tok_s)
    return TrialResult(config=Config(backend="vllm", quant="bf16", batch=batch), stage="batch",
                       metrics=best.model_copy(update={"by_concurrency": by}))


# L4, Qwen2.5-7B, sharegpt: at 32 users the median met the 1000 ms ceiling and the tail did not.
L4_SHAREGPT = _trial(64, {8: (212.0, 400.0, 780.0), 32: (735.0, 989.0, 1222.0)})


def test_the_tail_decides_by_default():
    winner, notes = pick([L4_SHAREGPT], "balanced", Constraints(ttft_ceiling_ms=1000))
    assert any("concurrency 8: 212.0 tok/s" in n for n in notes) and "(p95 780)" in " ".join(notes)


def test_the_median_is_still_available():
    _, notes = pick([L4_SHAREGPT], "balanced", Constraints(ttft_ceiling_ms=1000, ttft_percentile=50))
    assert any("concurrency 32: 735.0 tok/s" in n for n in notes)


def test_a_trial_without_a_recorded_p95_falls_back_to_the_median():
    t = _trial(64, {8: (212.0, 400.0, None), 32: (735.0, 989.0, None)})
    _, notes = pick([t], "balanced", Constraints(ttft_ceiling_ms=1000))
    assert any("concurrency 32" in n for n in notes)


def test_profiles_picked_by_the_median_are_cached_apart():
    assert SearchOptions().key("balanced") == {"ttft": "p95"}
    assert SearchOptions(ttft_percentile=50).key("balanced") == {}  # what profiles from before the option recorded
    assert SearchOptions().key("throughput") == {}  # no TTFT ceiling, so nothing to recalibrate


def test_only_the_median_or_p95_is_accepted():
    assert _percentile(95) == 95 and _percentile(50) == 50
    with pytest.raises(typer.BadParameter):
        _percentile(99)
