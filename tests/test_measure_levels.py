"""Trials must not measure a prefix cache they filled themselves, nor a load generator's ceiling."""

from __future__ import annotations

import polyserve.calibrate.measure as M
from polyserve.backends.base import LlmtraceHooks
from polyserve.calibrate.objectives import Constraints, enough_level
from polyserve.calibrate.tokens import TokenCounter
from polyserve.calibrate.workload import Workload
from polyserve.models import TrialMetrics


def _words(text: str) -> int:
    return len(text.split())


def test_each_concurrency_level_gets_fresh_prompts(monkeypatch):
    seen = []

    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        seen.append((concurrency, list(workload.prompts)))
        return [M.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.05, tokens=4, token_source="usage")]

    monkeypatch.setattr(M, "_drive", fake_drive)
    wl = Workload(n_prompts=3, prefill_tokens=64, shared_prefix_tokens=32, decode_tokens=4, concurrencies=(1, 2, 4))
    M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(encode=_words, name="words"))
    assert [c for c, _ in seen] == [1, 2, 4]
    levels = [set(p) for _, p in seen]
    assert not (levels[0] & levels[1] or levels[1] & levels[2] or levels[0] & levels[2])
    assert all(p.startswith(wl.prefix_text) for s in levels for p in s)  # the shared prefix is kept
    assert all(abs(_words(p) - 64) <= 2 for s in levels for p in s)  # every level fitted to length


def _timed_drive(seen, ttft_s):
    """A fake load generator whose time to first token depends on the concurrency level."""
    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        seen.append((concurrency, tuple(workload.prompts)))
        t = ttft_s[concurrency]
        return [M.RequestOutcome(ok=True, ttft_s=t, duration_s=t + 0.05, tokens=4, token_source="usage")] * 8
    return fake_drive


def test_levels_run_highest_first_and_stop_once_one_meets_the_objective(monkeypatch):
    seen = []
    monkeypatch.setattr(M, "_drive", _timed_drive(seen, {1: 0.1, 4: 0.3, 8: 0.9}))
    wl = Workload(n_prompts=3, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4, 8))
    enough = enough_level("balanced", Constraints(ttft_ceiling_ms=500))
    m = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(), enough=enough)
    assert [c for c, _ in seen] == [8, 4]  # 8 users broke the 500 ms ceiling; 4 met it, so 1 cannot score higher
    assert sorted(m.by_concurrency, key=int) == ["4", "8"]


def test_a_level_keeps_its_prompts_whatever_the_order(monkeypatch):
    bottom_up, top_down = [], []
    wl = Workload(n_prompts=3, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4, 8))
    for seen, enough in ((bottom_up, None), (top_down, lambda m: False)):
        monkeypatch.setattr(M, "_drive", _timed_drive(seen, {1: 0.1, 4: 0.3, 8: 0.9}))
        M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(), enough=enough)
    assert [c for c, _ in bottom_up] == [1, 4, 8] and [c for c, _ in top_down] == [8, 4, 1]
    assert dict(bottom_up) == dict(top_down)


def test_objectives_that_can_prefer_a_lower_level_measure_every_one():
    assert enough_level("latency") is None and enough_level("efficiency") is None
    fast = TrialMetrics(tok_s=900, ttft_ms=100, ttft_p95_ms=700, tpot_ms=20, requests=8, output_tokens=64, concurrency=8)
    assert enough_level("throughput")(fast)
    assert not enough_level("balanced", Constraints(ttft_ceiling_ms=500))(fast)  # the tail breaks the ceiling
    assert enough_level("balanced", Constraints(ttft_ceiling_ms=500, ttft_percentile=50))(fast)
    assert not enough_level("throughput", Constraints(tpot_ceiling_ms=10))(fast)  # the decode ceiling applies too


def test_failed_requests_keep_their_errors():
    ok = M.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.05, tokens=4, token_source="usage")
    outcomes = [M.RequestOutcome(ok=False, error="HTTP 500: boom"), M.RequestOutcome(ok=False, error="HTTP 500: boom"),
                M.RequestOutcome(ok=False, error="timeout"), ok]
    m = M._metrics_from(outcomes, 1.0, 4)
    assert m.errors == ["HTTP 500: boom", "timeout"]
    assert m.failure_summary() == "3/4 requests failed: HTTP 500: boom"
    assert M._metrics_from([ok], 1.0, 1).errors == []


def test_several_client_processes_share_one_level():
    from tests.test_measure_and_proxy import _Server
    from tests.test_strategies import _counting_engine

    app = _counting_engine("a")
    wl = Workload(n_prompts=6, prefill_tokens=16, decode_tokens=2, concurrencies=(4,))
    with _Server(app) as s:
        outcomes, wall = M._drive_clients(s.url, LlmtraceHooks(), wl, 4, 30.0, clients=3)
    assert len(outcomes) == 6 and all(o.ok for o in outcomes) and wall > 0
    assert app.state.count == 6  # every prompt sent once, across the three processes
