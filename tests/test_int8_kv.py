"""vLLM's int8 KV cache with a scale per token and head, run through Triton attention."""

from __future__ import annotations

import importlib.util

import polyserve.backends.vllm as vl
from polyserve.backends import get_backend
from polyserve.bench.ablation import strategy_variants
from polyserve.calibrate.workload import get_workload
from polyserve.hfconfig import kv_element_bytes
from polyserve.models import Config
from tests.conftest import make_hw


def _flashinfer(monkeypatch, present: bool) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if present else None) if name == "flashinfer" else real(name, *a))


def test_offered_from_vllm_029_on_ampere_and_newer(monkeypatch, hw_a100):
    vllm = get_backend("vllm")
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 29))
    _flashinfer(monkeypatch, False)
    assert vllm.kv_dtypes(hw_a100) == [vl.INT8_KV]  # Ampere without FlashInfer now has a quantized cache
    _flashinfer(monkeypatch, True)
    assert vllm.kv_dtypes(hw_a100) == [vl.AMPERE_FP8_KV, vl.INT8_KV]
    assert vllm.kv_dtypes(make_hw("rtx4090")) == ["fp8", vl.INT8_KV]
    assert vllm.kv_dtypes(make_hw("gtx1080")) == []
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 11))
    assert vllm.kv_dtypes(hw_a100) == [vl.AMPERE_FP8_KV]


def test_it_launches_with_triton_attention(monkeypatch, prepared_vllm):
    monkeypatch.setattr(vl, "_attention_backend_flag", lambda: True)
    spec = get_backend("vllm").launch_spec(Config(backend="vllm", quant="bf16", kv_dtype=vl.INT8_KV), prepared_vllm, 1)
    assert spec.args[spec.args.index("--kv-cache-dtype") + 1] == "int8_per_token_head"
    assert spec.args[spec.args.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert spec.env["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"


def test_the_planner_sizes_it_between_fp8_and_bf16():
    assert kv_element_bytes("fp8_e5m2") < kv_element_bytes(vl.INT8_KV) < kv_element_bytes("bfloat16")


def test_the_ablation_keeps_triton_attention_without_the_int8_cache(hw_a100, prepared_vllm):
    pick = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9, kv_dtype=vl.INT8_KV)
    kv = {v.label: v.config for v in strategy_variants(pick, get_backend("vllm"), hw_a100, prepared_vllm,
                                                       get_workload("chat")) if v.strategy == "kv"}
    assert set(kv) == {"-kv", "-kv (Triton kept)"}
    assert kv["-kv"].kv_dtype == "auto" and "attention_backend" not in kv["-kv"].extra
    assert kv["-kv (Triton kept)"].extra["attention_backend"] == "TRITON_ATTN"
