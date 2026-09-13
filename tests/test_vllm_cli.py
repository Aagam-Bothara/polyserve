"""vLLM launch command across versions: `vllm serve` where installed, --attention-backend where known."""

from __future__ import annotations

import polyserve.backends.vllm as vl
from polyserve.backends import get_backend
from polyserve.models import Config


def test_vllm_serve_cli_when_installed(prepared_vllm, monkeypatch):
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=16)
    monkeypatch.setattr(vl, "_serve_cli", lambda: "/usr/local/bin/vllm")
    args = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1).args
    assert args[:3] == ["/usr/local/bin/vllm", "serve", prepared_vllm.spec.hf_id] and "--model" not in args
    monkeypatch.setattr(vl, "_serve_cli", lambda: None)  # no CLI: the older module
    legacy = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1).args
    assert legacy[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert legacy[legacy.index("--model") + 1] == prepared_vllm.spec.hf_id


def test_fp8_is_not_offered_on_ampere_with_vllm_029(hw_a100, hw_rtx4090, monkeypatch):
    be = get_backend("vllm")
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 29))
    assert "fp8" not in be.precisions(hw_a100)  # Ampere: broken upstream in 0.29
    assert "fp8" in be.precisions(hw_rtx4090)  # Ada: native fp8, unaffected
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 11))
    assert "fp8" in be.precisions(hw_a100)  # 0.11 runs it through Marlin


def test_attention_backend_flag_only_where_vllm_has_it(prepared_vllm, monkeypatch):
    cfg = Config(backend="vllm", quant="bf16", kv_dtype=vl.AMPERE_FP8_KV)
    monkeypatch.setattr(vl, "_attention_backend_flag", lambda: False)  # vLLM 0.11
    old = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1)
    assert old.env["VLLM_ATTENTION_BACKEND"] == "FLASHINFER" and "--attention-backend" not in old.args
    monkeypatch.setattr(vl, "_attention_backend_flag", lambda: True)  # newer vLLM
    new = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1)
    assert new.args[new.args.index("--attention-backend") + 1] == "FLASHINFER"
    plain = get_backend("vllm").launch_spec(Config(backend="vllm", quant="bf16"), prepared_vllm, 1)
    assert "--attention-backend" not in plain.args and "VLLM_ATTENTION_BACKEND" not in plain.env
