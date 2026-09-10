from __future__ import annotations

import pytest

from polyserve import cache
from polyserve.hardware import hardware_hash
from polyserve.models import Config, Profile
from polyserve.pipeline import calibrate, default_profile, prepare_and_plan, resolve_profile, select
from tests.test_objectives_and_search import FakeRunner


def _profile(hw, spec, prepared) -> Profile:
    return Profile(
        polyserve_version="0.1.0", hardware_hash=hardware_hash(hw), hardware=hw, model_id=spec.hf_id,
        objective="balanced", backend="vllm", backend_version="1.0",
        config=Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9),
        prepared=prepared, launch_args=["python", "-m", "vllm"],
    )


def test_cache_roundtrip(tmp_home, hw_a100, spec, prepared_vllm):
    p = _profile(hw_a100, spec, prepared_vllm)
    path = cache.save(p)
    assert path == tmp_home / "profiles" / hardware_hash(hw_a100) / spec.safe_id / "balanced.json"
    loaded = cache.load(hw_a100, spec, "balanced")
    assert loaded is not None and loaded.config.key() == p.config.key()
    assert loaded.prepared.arch.num_layers == 28
    assert cache.load(hw_a100, spec, "latency") is None
    assert len(cache.list_profiles()) == 1


def test_cache_invalidates_on_hardware_or_version_change(tmp_home, hw_a100, hw_rtx4090, spec, prepared_vllm):
    cache.save(_profile(hw_a100, spec, prepared_vllm))
    assert cache.load(hw_rtx4090, spec, "balanced") is None  # different hash -> different path anyway
    hw_a100.backends["vllm"].version = "2.0"
    assert cache.load(hw_a100, spec, "balanced") is None
    hw_a100.backends["vllm"].version = "1.0"
    hw_a100.backends["vllm"].available = False
    assert cache.load(hw_a100, spec, "balanced") is None


def test_cache_delete(tmp_home, hw_a100, spec, prepared_vllm):
    cache.save(_profile(hw_a100, spec, prepared_vllm))
    assert cache.delete(hw_a100, spec) == 1
    assert cache.load(hw_a100, spec, "balanced") is None


@pytest.mark.usefixtures("no_network")
def test_prepare_and_plan_all_candidates(hw_a100, spec):
    candidates, reg = select(hw_a100, spec)
    result = prepare_and_plan(hw_a100, spec, candidates, reg)
    assert set(result.feasible) == {"vllm", "sglang", "llamacpp-cuda"}
    assert result.errors == {}
    assert result.total_considered > len(result.all_feasible) > 0


@pytest.mark.usefixtures("no_network")
def test_default_profile_uses_first_candidate(hw_a100, spec):
    candidates, reg = select(hw_a100, spec)
    result = prepare_and_plan(hw_a100, spec, candidates, reg)
    p = default_profile(hw_a100, spec, result, reg)
    assert p.backend == "vllm" and "uncalibrated defaults" in p.notes
    assert "--max-num-seqs" in p.launch_args


@pytest.mark.usefixtures("no_network")
def test_calibrate_produces_profile_with_table(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    result = prepare_and_plan(hw_a100, spec, candidates, reg)
    p = calibrate(hw_a100, spec, "throughput", result, reg, runner=FakeRunner())
    assert p.backend == "vllm" and p.config.quant == "fp8"
    assert len(p.calibration_table) > 3
    assert any("calibration took" in n for n in p.notes)


@pytest.mark.usefixtures("no_network")
def test_resolve_profile_caches_then_hits(tmp_home, hw_a100, spec, monkeypatch):
    import polyserve.pipeline as pl

    calls = {"n": 0}
    runner = FakeRunner()

    def fake_calibrate(hw, spec_, objective, plan_, reg, **kw):
        calls["n"] += 1
        return calibrate(hw, spec_, objective, plan_, reg, runner=runner)

    monkeypatch.setattr(pl, "calibrate", fake_calibrate)
    stages = []
    p1 = resolve_profile(spec, "throughput", force_backend="vllm", hw=hw_a100, on_stage=stages.append)
    assert calls["n"] == 1 and "calibrate" in stages
    p2 = resolve_profile(spec, "throughput", force_backend="vllm", hw=hw_a100, on_stage=stages.append)
    assert calls["n"] == 1 and "cache hit" in stages and p2.config.key() == p1.config.key()
    resolve_profile(spec, "throughput", force_backend="vllm", hw=hw_a100, recalibrate=True)
    assert calls["n"] == 2


@pytest.mark.usefixtures("no_network")
def test_resolve_profile_skip_calibration(tmp_home, hw_cpu, spec, monkeypatch):
    import polyserve.backends.llamacpp as lc

    def fake_materialize(self, m, quants):
        for q in quants:
            m.gguf_paths[q] = f"/models/{q}.gguf"
        return m

    monkeypatch.setattr(lc.LlamaCppBackend, "materialize", fake_materialize)
    p = resolve_profile(spec, hw=hw_cpu, skip_calibration=True)
    assert p.launch_args[p.launch_args.index("-m") + 1] == f"/models/{p.config.quant}.gguf"
    assert p.backend == "llamacpp-cpu" and p.calibration_table == []
    assert cache.list_profiles() == []  # defaults are not cached


def test_select_with_no_backends(hw_cpu, spec):
    for b in hw_cpu.backends.values():
        b.available = False
    candidates, _ = select(hw_cpu, spec)
    assert candidates == []
