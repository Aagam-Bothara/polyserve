"""vLLM's GPU n-gram lookup and suffix decoding as speculative-decoding candidates."""

from __future__ import annotations

import importlib.util

import pytest

from polyserve import speculative


def test_the_new_methods_become_vllm_speculative_configs():
    assert speculative.vllm_config("ngram_gpu:4") == {
        "method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    assert speculative.vllm_config(speculative.suffix()) == {"method": "suffix", "num_speculative_tokens": 24}
    assert speculative.parse("suffix") == ("suffix", None, 24) and speculative.parse("ngram_gpu") == ("ngram_gpu", None, 4)
    with pytest.raises(ValueError):
        speculative.parse("eagle3:4")


def test_versions_gate_them():
    assert speculative.vllm_supports_ngram_gpu("0.29.0") and not speculative.vllm_supports_ngram_gpu("0.28.1")
    assert not speculative.vllm_supports_suffix("0.28.1")


def test_suffix_decoding_needs_arctic_inference(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "arctic_inference" else real(name, *a))
    assert not speculative.vllm_supports_suffix("0.29.0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "arctic_inference" else real(name, *a))
    assert speculative.vllm_supports_suffix("0.29.0")


@pytest.mark.parametrize("version, arctic, expected", [
    ("0.29.0", True, ["ngram_gpu:4", "suffix:24", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]),
    ("0.29.0", False, ["ngram_gpu:4", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]),
    ("0.28.1", True, ["ngram:4", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]),
    ("0.11.0", False, ["ngram:4"]),  # no separate draft model before 0.12
])
def test_vllm_offers_what_the_installed_release_supports(monkeypatch, prepared_vllm, version, arctic, expected):
    from polyserve.backends import get_backend
    from polyserve.models import Config

    monkeypatch.setattr(speculative, "_vllm_version", lambda: version)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if arctic else None) if name == "arctic_inference" else real(name, *a))
    cfg = Config(backend="vllm", quant="bf16", batch=64)
    model = prepared_vllm.model_copy(update={"spec": prepared_vllm.spec.model_copy(update={"hf_id": "Qwen/Qwen2.5-3B-Instruct"})})
    assert [c.spec_decode for c in get_backend("vllm").spec_variants(cfg, model)] == expected
