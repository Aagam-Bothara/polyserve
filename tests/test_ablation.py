"""The strategy ablation: variants of the pick, result loading, sweeps and crossover."""

from __future__ import annotations

import json

import pytest

from polyserve.backends import get_backend
from polyserve.bench.ablation import (
    crossover, load_pick, markdown, score_row, strategy_variants, sweep_workload,
)
from polyserve.bench.compare import results_path
from polyserve.calibrate.objectives import Constraints
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config, ModelSpec, TrialMetrics, TrialResult


@pytest.fixture
def drafts(monkeypatch):
    import polyserve.backends.vllm as vl
    import polyserve.speculative as S

    monkeypatch.setattr(S, "vllm_supports_draft", lambda version=None: True)
    real = vl.importlib.util.find_spec
    monkeypatch.setattr(vl.importlib.util, "find_spec", lambda n: object() if n == "flashinfer" else real(n))


@pytest.mark.usefixtures("drafts")
def test_variants_flip_each_strategy(hw_a100, prepared_vllm):
    be = get_backend("vllm")
    model = prepared_vllm.model_copy(update={"weights_bytes": {**prepared_vllm.weights_bytes, "awq": 2_000_000_000},
                                             "hf_paths": {"awq": "org/Llama-3.2-3B-Instruct-AWQ"}})
    pick = Config(backend="vllm", quant="fp8", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    labels = [v.label for v in strategy_variants(pick, be, hw_a100, model, get_workload("chat-system"))]
    assert labels == ["+awq", "+kv:fp8_e5m2", "-prefix", "+spec:ngram:4",
                      "+spec:draft:meta-llama/Llama-3.2-1B-Instruct:4"]
    no_prefix = [v.label for v in strategy_variants(pick, be, hw_a100, model, get_workload("chat"))]
    assert "-prefix" not in no_prefix  # nothing shared, nothing for a prefix cache to do

    used = pick.model_copy(update={"quant": "awq", "kv_dtype": "fp8", "spec_decode": "ngram:4"})
    vs = {v.label: v.config for v in strategy_variants(used, be, hw_a100, model, get_workload("chat"))}
    assert set(vs) == {"-int4 (fp8)", "-kv", "-spec"}
    assert vs["-int4 (fp8)"].quant == "fp8" and vs["-kv"].kv_dtype == "auto" and vs["-spec"].spec_decode is None


def test_removing_kv_quant_steps_the_batch_down_to_fit(hw_gtx1080, prepared_llamacpp):
    pick = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=16, n_gpu_layers=29, n_batch=512,
                  kv_dtype="q4_0")
    (kv,) = [v for v in strategy_variants(pick, get_backend("llamacpp-cuda"), hw_gtx1080, prepared_llamacpp,
                                          get_workload("chat")) if v.strategy == "kv"]
    assert kv.label == "-kv" and kv.config.kv_dtype == "auto" and kv.config.batch == 8
    assert "does not fit at batch 16" in kv.note


def test_llamacpp_prefix_flags_are_added_or_removed(hw_gtx1080, prepared_llamacpp):
    be = get_backend("llamacpp-cuda")
    pick = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=29, n_batch=512)
    shared = get_workload("chat-system")
    added = [v.label for v in strategy_variants(pick, be, hw_gtx1080, prepared_llamacpp, shared)
             if v.strategy == "prefix"]
    assert added == ["+prefix:cache_reuse", "+prefix:kv_unified"]
    tuned = pick.model_copy(update={"extra": {"kv_unified": True}})
    (removed,) = [v for v in strategy_variants(tuned, be, hw_gtx1080, prepared_llamacpp, shared)
                  if v.strategy == "prefix"]
    assert removed.label == "-prefix" and removed.config.extra == {}


def test_load_pick_prefers_this_machine(tmp_path):
    spec = ModelSpec(hf_id="org/M")

    def write(hw_hash: str, quant: str) -> None:
        rows = [{"label": "vllm-default", "config": Config(backend="vllm", quant="bf16").model_dump()},
                {"label": "polyserve", "config": Config(backend="vllm", quant=quant).model_dump()}]
        results_path(hw_hash, spec, "chat", "balanced", tmp_path).write_text(json.dumps({"rows": rows}))

    write("aaaa", "bf16")
    write("zzzz", "fp8")
    assert load_pick(tmp_path, "org/M", "chat", "balanced", hw_hash="zzzz").quant == "fp8"
    assert load_pick(tmp_path, "org/M", "chat", "balanced").quant == "bf16"
    assert load_pick(tmp_path, "org/M", "rag", "balanced") is None


def test_sweep_workload_and_crossover():
    base = get_workload("rag-shared")
    wl = sweep_workload(base, [8, 1, 64, 4])
    assert wl.concurrencies == (1, 4, 8, 64) and wl.n_prompts == 128 and len(wl.prompts) == 128
    assert wl.prefix_text == base.prefix_text and all(p.startswith(wl.prefix_text) for p in wl.prompts)

    off = {"1": {"ok": True, "tok_s": 60, "e2e_ms": 2000}, "8": {"ok": True, "tok_s": 400, "e2e_ms": 2500},
           "32": {"ok": True, "tok_s": 1200, "e2e_ms": 3500}}
    on = {"1": {"ok": True, "tok_s": 110, "e2e_ms": 1200}, "8": {"ok": True, "tok_s": 390, "e2e_ms": 2400},
          "32": {"ok": True, "tok_s": 900, "e2e_ms": 4800}}
    assert crossover(off, on, "tok_s") == 8
    assert crossover(off, on, "e2e_ms") == 32
    assert crossover(off, {"1": {"ok": True, "tok_s": 999, "e2e_ms": 1}}) is None


def test_score_row_and_markdown():
    def result(tok_s: float, cfg: Config) -> TrialResult:
        level = TrialMetrics(tok_s=tok_s, ttft_ms=30, tpot_ms=10, requests=16, output_tokens=2048, concurrency=8)
        return TrialResult(config=cfg, stage="t", metrics=level.model_copy(update={"by_concurrency": {"8": level}}))

    cons = Constraints(ttft_ceiling_ms=500)
    base = Config(backend="vllm", quant="fp8")
    rows = [{"label": label, "config_key": cfg.key(), **score_row(result(tok, cfg), "balanced", cons)}
            for label, tok, cfg in (("polyserve", 500, base), ("+kv:fp8", 550, base.model_copy(update={"kv_dtype": "fp8"})))]
    rows.append({"label": "-kv", "config_key": "-", "ok": False, "note": "does not fit in memory"})
    top = rows[0]
    assert top["ok"] and top["concurrency"] == 8 and top["meets_slo"]
    assert top["e2e_ms"] == pytest.approx(30 + 10 * 127) and top["levels"]["8"]["tok_s"] == 500
    md = markdown(rows)
    assert "+10.0%" in md and "does not fit in memory" in md
    failed = score_row(TrialResult(config=base, stage="t", metrics=TrialMetrics(), error="boom"), "balanced", cons)
    assert failed == {"ok": False, "error": "boom"}
