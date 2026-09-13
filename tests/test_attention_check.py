"""On Ampere the fp8 KV cache also moves vLLM's attention to FlashInfer; the ablation separates the two."""

from __future__ import annotations

import polyserve.backends.vllm as vl
from polyserve.backends import get_backend
from polyserve.bench.ablation import strategy_variants
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config


def test_forced_attention_backend_is_set_once_and_not_rendered_as_an_extra(prepared_vllm, monkeypatch):
    cfg = Config(backend="vllm", quant="bf16", extra={"attention_backend": "FLASHINFER"})
    monkeypatch.setattr(vl, "_attention_backend_flag", lambda: True)
    spec = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1)
    assert spec.env["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"
    assert spec.args.count("--attention-backend") == 1 and "--kv-cache-dtype" not in spec.args
    monkeypatch.setattr(vl, "_attention_backend_flag", lambda: False)  # vLLM 0.11: the variable only
    old = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1)
    assert old.env["VLLM_ATTENTION_BACKEND"] == "FLASHINFER" and "--attention-backend" not in old.args


def test_ampere_fp8_cache_ablation_also_keeps_flashinfer_without_it(hw_a100, prepared_vllm):
    pick = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9,
                  kv_dtype=vl.AMPERE_FP8_KV)
    kv = {v.label: v.config for v in strategy_variants(pick, get_backend("vllm"), hw_a100, prepared_vllm,
                                                       get_workload("chat")) if v.strategy == "kv"}
    assert set(kv) == {"-kv", "-kv (FlashInfer kept)"}
    assert kv["-kv"].kv_dtype == "auto" and "attention_backend" not in kv["-kv"].extra
    kept = kv["-kv (FlashInfer kept)"]
    assert kept.kv_dtype == "auto" and kept.extra["attention_backend"] == "FLASHINFER"
