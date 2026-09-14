"""`--quant auto` leaves out 4-bit checkpoints; they are opt-in."""

from __future__ import annotations

import pytest

import polyserve.quantized as Q
from polyserve.backends import registry
from polyserve.models import Config
from polyserve.pipeline import SearchOptions, _usable, allowed_quants, prepare_and_plan


def test_auto_skips_4bit_checkpoints_unless_asked():
    supported = ["bf16", "fp8", "awq", "gptq"]
    assert allowed_quants(supported, None) == ["bf16", "fp8"]
    assert allowed_quants(supported, ["auto", "awq"]) == ["bf16", "fp8", "awq"]
    assert allowed_quants(supported, ["gptq"]) == ["gptq"]
    assert allowed_quants(supported, ["Q4_K_M"]) == []
    assert allowed_quants(["Q4_K_M", "Q8_0"], None) == ["Q4_K_M", "Q8_0"]  # llama.cpp keeps its formats
    assert allowed_quants([], None) is None  # a backend that lists nothing decides for itself


@pytest.mark.usefixtures("no_network")
def test_planner_prepares_4bit_only_on_request(hw_a100, spec, monkeypatch):
    monkeypatch.setattr(Q, "find_prequantized_repos", lambda s, methods=Q.INT4_METHODS, **kw: {
        m: Q.PrequantizedRepo(repo_id=f"org/Llama-3.2-3B-Instruct-{m.upper()}", method=m, size_bytes=2_000_000_000)
        for m in methods})
    reg = registry()
    default = prepare_and_plan(hw_a100, spec, ["vllm"], reg)
    assert set(default.prepared["vllm"].weights_bytes) == {"bf16", "fp8"}
    opted = prepare_and_plan(hw_a100, spec, ["vllm"], reg, quants=["auto", "awq"])
    assert set(opted.prepared["vllm"].weights_bytes) == {"bf16", "fp8", "awq"}
    only = prepare_and_plan(hw_a100, spec, ["vllm"], reg, quants=["gptq"])
    assert set(only.prepared["vllm"].weights_bytes) == {"gptq"}


def test_options_and_cached_profiles_follow_the_default(hw_a100, spec, prepared_vllm):
    from tests.test_cache_and_pipeline import _profile

    default, opted = SearchOptions(), SearchOptions(quants=["auto", "gptq"])
    assert default.allows("fp8") and default.allows("Q4_K_M") and not default.allows("gptq")
    assert opted.allows("gptq") and opted.allows("bf16") and not opted.allows("awq")
    assert default.key() == {} and opted.key() == {"quant": "auto+gptq"}

    base = _profile(hw_a100, spec, prepared_vllm)
    gptq = base.model_copy(update={"config": Config(backend="vllm", quant="gptq")})
    assert not _usable(gptq, default, None)  # calibrated when auto still picked 4-bit: recalibrate
    assert _usable(gptq, opted, None)
    assert _usable(base, default, None) and not _usable(base, default, "llamacpp-cuda")
    assert not _usable(None, default, None)


def test_cli_accepts_auto_inside_a_list():
    from polyserve.cli import _opts, _quant_list

    assert _quant_list("auto,awq") == "auto,awq"
    assert _opts("auto,awq", "on", "on", "on").quants == ["auto", "awq"]
    assert _opts("auto", "on", "on", "on").quants is None
