from __future__ import annotations

import pytest
from typer.testing import CliRunner

from polyserve import cache
from polyserve.backends import get_backend
from polyserve.backends.base import ctx_grid
from polyserve.calibrate.workload import WORKLOAD_NAMES, Workload, get_workload, workload_table
from polyserve.cli import app
from polyserve.memory import plan
from polyserve.pipeline import default_profile, prepare_and_plan, select


def test_presets_are_sane():
    assert set(WORKLOAD_NAMES) == {"default", "chat", "long-context", "generation", "high-concurrency", "rag",
                                   "chat-system", "rag-shared", "sharegpt", "extract", "code-edit"}
    for name in WORKLOAD_NAMES:
        w = get_workload(name)
        assert w.name == name
        # Real-text presets download their prompts when a trial needs them, not before.
        assert len(w.prompts) == (w.n_prompts if w.source == "synthetic" else 0)
        assert w.n_prompts >= max(w.concurrencies)  # every concurrency level can actually be reached
        assert w.min_ctx > w.prefill_tokens + w.decode_tokens
    assert get_workload("default").spec() == Workload().spec()
    with pytest.raises(ValueError):
        get_workload("batch-inference")


def test_get_workload_returns_independent_copies():
    a = get_workload("chat")
    b = get_workload("chat")
    a.prompts[0] = "changed"
    a.fitted = True
    assert b.prompts[0] != "changed" and not b.fitted


def test_ctx_grid_respects_workload_minimum():
    assert ctx_grid(131072, 0) == [2048, 4096, 8192]  # default workload: the spec's grid
    assert ctx_grid(4096, 0) == [2048, 4096]
    assert ctx_grid(131072, get_workload("long-context").min_ctx) == [16384, 32768, 65536]
    assert ctx_grid(131072, get_workload("rag").min_ctx) == [8192, 16384, 32768]
    assert ctx_grid(32768, get_workload("long-context").min_ctx) == [16384, 32768]
    assert ctx_grid(9000, 8500) == [9000]  # only the model max fits the workload
    assert ctx_grid(4096, 8500) == []  # model cannot run the workload at all


def test_long_context_planning_only_keeps_large_contexts(hw_a100, prepared_vllm):
    be = get_backend("vllm")
    wl = get_workload("long-context")
    grid = be.candidate_configs(hw_a100, prepared_vllm, min_ctx=wl.min_ctx)
    assert grid and all(c.ctx >= wl.min_ctx for c in grid)
    kept = plan(hw_a100, prepared_vllm, grid, be.memory_model(hw_a100))
    assert kept and all(c.ctx >= wl.min_ctx for c, _ in kept)
    assert be.default_config(hw_a100, prepared_vllm, min_ctx=wl.min_ctx).ctx >= wl.min_ctx


@pytest.mark.usefixtures("no_network")
def test_prepare_and_plan_rejects_impossible_workload(hw_a100, spec, llama3b_arch):
    llama3b_arch.max_position_embeddings = 4096
    candidates, reg = select(hw_a100, spec, force="vllm")
    result = prepare_and_plan(hw_a100, spec, candidates, reg, workload=get_workload("long-context"))
    assert "vllm" in result.errors and "workload minimum" in result.errors["vllm"]


@pytest.mark.usefixtures("no_network")
def test_default_profile_records_workload(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    wl = get_workload("rag")
    result = prepare_and_plan(hw_a100, spec, candidates, reg, workload=wl)
    p = default_profile(hw_a100, spec, result, reg, workload=wl)
    assert p.workload == "rag" and p.workload_spec["prefill_tokens"] == 6144
    assert p.config.ctx >= wl.min_ctx


def test_cache_is_keyed_by_workload(tmp_home, hw_a100, spec, prepared_vllm):
    from tests.test_cache_and_pipeline import _profile

    p = _profile(hw_a100, spec, prepared_vllm)
    cache.save(p)
    q = p.model_copy(update={"workload": "chat"})
    path = cache.save(q)
    assert path.name == "balanced-chat.json"
    assert cache.load(hw_a100, spec, "balanced").workload == "default"
    assert cache.load(hw_a100, spec, "balanced", "chat").workload == "chat"
    assert cache.load(hw_a100, spec, "balanced", "rag") is None
    assert len(cache.list_profiles()) == 2


def test_workloads_command_and_option_validation():
    runner = CliRunner()
    r = runner.invoke(app, ["workloads"])
    assert r.exit_code == 0 and "long-context" in r.output and "6144" in r.output
    assert len(workload_table()) == len(WORKLOAD_NAMES)
    r = runner.invoke(app, ["bench", "x/y", "--workload", "batch"])
    assert r.exit_code != 0
