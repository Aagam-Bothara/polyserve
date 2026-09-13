"""Workloads built from real text: sampled from a source, capped in length, answers stop naturally."""

from __future__ import annotations

import asyncio

import pytest

import polyserve.calibrate.datasets as D
import polyserve.calibrate.measure as M
from polyserve.backends.base import LlmtraceHooks
from polyserve.calibrate.tokens import TokenCounter
from polyserve.calibrate.workload import WORKLOAD_NAMES, Workload, get_workload


@pytest.fixture
def fake_sources(monkeypatch):
    articles = [(f"article {i}: " + "word " * (20 + i), D.EXTRACT_INSTRUCTION) for i in range(100)]
    monkeypatch.setitem(D.LOADERS, "cnn-extract", lambda: list(articles))
    monkeypatch.setitem(D.LOADERS, "humaneval-edit", lambda: [(f"def f{i}():\n    return {i}", D.EDIT_INSTRUCTION)
                                                             for i in range(5)])
    D.pool.cache_clear()
    yield
    D.pool.cache_clear()


def test_presets_do_not_download_until_prompts_are_needed(monkeypatch):
    def boom():
        raise AssertionError("downloaded before any prompt was needed")

    monkeypatch.setitem(D.LOADERS, "cnn-extract", boom)
    D.pool.cache_clear()
    wl = get_workload("extract")
    assert wl.prompts == [] and wl.natural_stop and wl.source == "cnn-extract"
    assert {"sharegpt", "extract", "code-edit"} <= set(WORKLOAD_NAMES)
    assert "cnn-extract prompts of up to 1536 tokens" in wl.describe() and wl.spec()["natural_stop"] is True
    D.pool.cache_clear()


@pytest.mark.usefixtures("fake_sources")
def test_prompts_come_from_the_source_capped_and_disjoint_by_offset():
    wl = get_workload("extract").ensure_prompts()
    assert len(wl.prompts) == 32 and all(p.endswith(D.EXTRACT_INSTRUCTION) for p in wl.prompts)
    capped = Workload(name="x", source="cnn-extract", n_prompts=4, prefill_tokens=20).ensure_prompts()
    assert all(len(p) <= 64 + len(D.EXTRACT_INSTRUCTION) for p in capped.prompts)  # body cut, instruction kept
    a = Workload(name="x", source="cnn-extract", n_prompts=10).ensure_prompts()
    b = Workload(name="x", source="cnn-extract", n_prompts=10, sample_offset=10).ensure_prompts()
    assert not set(a.prompts) & set(b.prompts)
    assert len(Workload(name="y", source="humaneval-edit", n_prompts=7).ensure_prompts().prompts) == 7  # wraps


@pytest.mark.usefixtures("fake_sources")
def test_fit_trims_long_real_prompts_but_keeps_the_instruction():
    words = TokenCounter(encode=lambda t: len(t.split()), name="words")
    wl = Workload(name="x", source="cnn-extract", n_prompts=6, prefill_tokens=60).ensure_prompts()
    wl.fit_prompts(words)
    assert wl.fitted and all(len(p.split()) <= 60 and p.endswith(D.EXTRACT_INSTRUCTION) for p in wl.prompts)


@pytest.mark.usefixtures("fake_sources")
def test_each_level_draws_fresh_real_prompts(monkeypatch):
    seen = []

    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        seen.append(set(workload.prompts))
        return [M.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.05, tokens=4, token_source="usage")]

    monkeypatch.setattr(M, "_drive", fake_drive)
    wl = Workload(name="x", source="cnn-extract", natural_stop=True, n_prompts=5, concurrencies=(1, 4, 8))
    M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter())
    assert len(seen) == 3 and not (seen[0] & seen[1] or seen[1] & seen[2] or seen[0] & seen[2])


def test_natural_stop_lets_the_model_end_its_answer(monkeypatch):
    bodies = []

    async def fake_one(client, url, body, timeout, counter):
        bodies.append(body)
        return M.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.02, tokens=2, token_source="usage")

    monkeypatch.setattr(M, "_one_request", fake_one)
    for natural in (True, False):
        wl = Workload(n_prompts=1, prefill_tokens=16, decode_tokens=4, natural_stop=natural)
        asyncio.run(M._drive("http://x", LlmtraceHooks(), wl, 1, 5.0))
    assert [b["ignore_eos"] for b in bodies] == [False, True] and bodies[0]["max_tokens"] == 4
