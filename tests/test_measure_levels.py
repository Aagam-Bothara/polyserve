"""Trials must not measure a prefix cache they filled themselves, nor a load generator's ceiling."""

from __future__ import annotations

import polyserve.calibrate.measure as M
from polyserve.backends.base import LlmtraceHooks
from polyserve.calibrate.tokens import TokenCounter
from polyserve.calibrate.workload import Workload


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
