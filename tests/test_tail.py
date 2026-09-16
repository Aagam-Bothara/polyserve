"""A p95 from a few dozen requests gets a confidence interval, and a level too close to its ceiling is measured
again rather than decided by its second-slowest request."""

from __future__ import annotations

import math

import polyserve.calibrate.measure as M
from polyserve.backends.base import LlmtraceHooks
from polyserve.calibrate.objectives import Constraints, close_call_level
from polyserve.calibrate.tail import attained_confidence, quantile_interval, samples_needed
from polyserve.calibrate.tokens import TokenCounter
from polyserve.calibrate.workload import Workload
from polyserve.models import TrialMetrics


def test_the_interval_brackets_the_sample_percentile_and_narrows_with_more_samples():
    small = [float(i) for i in range(1, 33)]  # 32 requests, as at 8 users
    lo, hi = quantile_interval(small, 0.95, unbounded=False)
    assert lo <= small[round(0.95 * 31)] <= hi and hi == 32.0  # clamped: the top is the slowest request
    big = [float(i) for i in range(1, 641)]
    blo, bhi = quantile_interval(big, 0.95)
    assert blo <= 608 <= bhi and (bhi - blo) / 640 < (hi - lo) / 32  # narrower relative to the sample
    assert all(math.isnan(v) for v in quantile_interval([], 0.95))


def test_32_requests_cannot_bound_a_p95_at_95_percent():
    """The slowest of 32 samples exceeds the true p95 only 80.6% of the time, so it is not a 95% upper bound."""
    small = [float(i) for i in range(1, 33)]
    lo, hi = quantile_interval(small, 0.95)  # unbounded by default: say so rather than clamp
    assert hi == math.inf and lo <= 32.0
    assert round(attained_confidence(32, 0.95), 3) == 0.806
    assert samples_needed(0.95, 0.95, two_sided=True) == 72
    assert samples_needed(0.95, 0.95, two_sided=False) == 59


def test_enough_samples_give_a_finite_95_percent_upper_bound():
    plenty = [float(i) for i in range(1, 73)]  # 72 samples: the two-sided interval closes
    lo, hi = quantile_interval(plenty, 0.95)
    assert math.isfinite(lo) and math.isfinite(hi) and lo <= plenty[round(0.95 * 71)] <= hi
    assert attained_confidence(72, 0.95) >= 0.95


def _metrics(ttfts_ms) -> TrialMetrics:
    return TrialMetrics(requests=len(ttfts_ms), output_tokens=64 * len(ttfts_ms), tok_s=100.0, concurrency=8,
                        ttft_samples_ms=sorted(ttfts_ms))


def test_a_close_call_is_one_whose_ceiling_sits_inside_the_interval():
    spread = [300.0 + 50 * i for i in range(8)]  # 300 ... 650 ms: the p95's interval is 550-650 ms
    assert close_call_level("balanced", Constraints(ttft_ceiling_ms=600))(_metrics(spread))
    assert not close_call_level("balanced", Constraints(ttft_ceiling_ms=300))(_metrics(spread))  # clearly over
    assert not close_call_level("balanced", Constraints(ttft_ceiling_ms=2000))(_metrics(spread))  # clearly under
    assert close_call_level("throughput") is None  # no tail ceiling to be close to


def _spread_drive(seen):
    """A fake load generator: 8 requests per call, first tokens spread over 300-650 ms."""
    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        seen.append((concurrency, tuple(workload.prompts)))
        return [M.RequestOutcome(ok=True, ttft_s=0.30 + 0.05 * i, duration_s=1.0, tokens=64, token_source="usage")
                for i in range(8)]
    return fake_drive


def _run(monkeypatch, ceiling_ms):
    seen = []
    monkeypatch.setattr(M, "_drive", _spread_drive(seen))
    wl = Workload(n_prompts=8, prefill_tokens=16, decode_tokens=64, concurrencies=(8,))
    m = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(),
                    close_call=close_call_level("balanced", Constraints(ttft_ceiling_ms=ceiling_ms)))
    return m.by_concurrency["8"], seen


def test_a_close_level_is_measured_again_on_fresh_prompts_and_judged_on_both_runs(monkeypatch):
    level, seen = _run(monkeypatch, 600)
    assert len(seen) == 2 and not set(seen[0][1]) & set(seen[1][1])  # the second run has prompts of its own
    assert level.resampled and level.requests == 16 and len(level.ttft_samples_ms) == 16


def test_a_clear_level_is_measured_once(monkeypatch):
    level, seen = _run(monkeypatch, 2000)
    assert len(seen) == 1 and not level.resampled and level.ttft_samples_ms == [300.0 + 50 * i for i in range(8)]
