"""The markdown report built from compare, ablation and quality outputs."""

from __future__ import annotations

import json

from polyserve.bench.ablation import ablation_report, compare_table, gguf_quality_table, quality_table


def _row(label, key, tok, ok=True, slo=True):
    return {"label": label, "config_key": key, "ok": ok, "scored_tok_s": tok, "scored_ttft_ms": 50.0, "meets_slo": slo}


def test_compare_table_reports_gains_and_layout_notes(tmp_path):
    f = tmp_path / "h__Qwen__Qwen2.5-3B-Instruct__chat__balanced.json"
    f.write_text(json.dumps({
        "model_id": "Qwen/Qwen2.5-3B-Instruct", "workload": "chat",
        "notes": ["best multi-GPU layout 2 replicas: 7000 tok/s (+92.0% vs one GPU 3640); 2 replicas wins",
                  "calibration took 900s"],
        "rows": [_row("polyserve", "vllm/gptq/b64", 1000.0), _row("vllm-default", "vllm/bf16/b256", 500.0),
                 _row("vllm-fp8-default", "vllm/fp8/b256", 800.0, slo=False)],
    }))
    md = compare_table([f])
    assert "| Qwen2.5-3B-Instruct | chat | `vllm/gptq/b64` | 1000 |" in md
    assert "500 (+100%)" in md and "800 (+25%), missed SLO" in md
    assert "2 replicas wins" in md and "calibration took" not in md


def test_ablation_report_includes_the_sweep(tmp_path):
    lvl = lambda tok, e2e: {"ok": True, "tok_s": tok, "e2e_ms": e2e}  # noqa: E731
    f = tmp_path / "h__M__chat__ablation.json"
    f.write_text(json.dumps({
        "model_id": "org/M", "workload": "chat",
        "rows": [{"label": "polyserve", "config_key": "k", "ok": True, "tok_s": 1000.0, "ttft_ms": 50.0,
                  "tpot_ms": 7.0, "e2e_ms": 940.0, "meets_slo": True},
                 {"label": "+spec:ngram:4", "config_key": "k2", "ok": True, "tok_s": 880.0, "ttft_ms": 52.0,
                  "tpot_ms": 8.0, "e2e_ms": 1070.0, "meets_slo": True}],
        "spec_sweep": {"levels": [1, 8], "off": {"levels": {"1": lvl(120, 1000), "8": lvl(900, 1100)}},
                       "on": [{"spec": "ngram:4", "tok_s_crossover": 8, "latency_crossover": None,
                               "levels": {"1": lvl(150, 800), "8": lvl(850, 1050)}}]},
    }))
    md = ablation_report([f])
    assert "**M / chat**" in md and "-12.0%" in md
    assert "stops gaining at c=8" in md and "| 1 | 120 | 150 | 1000 ms | 800 ms |" in md


def test_quality_tables(tmp_path):
    q = tmp_path / "quality-3b.json"
    q.write_text(json.dumps({"model": "Qwen/Qwen2.5-3B-Instruct", "perplexity": {"bf16": 5.278, "awq": 6.647},
                             "delta_pct": {"bf16": 0.0, "awq": 25.9},
                             "checkpoints": {"awq": "Qwen/Qwen2.5-3B-Instruct-AWQ"}}))
    md = quality_table([q])
    assert "| Qwen2.5-3B-Instruct | awq | 6.647 | +25.9% | Qwen/Qwen2.5-3B-Instruct-AWQ |" in md
    g = tmp_path / "quality-gguf.txt"
    g.write_text("q8_0 Final estimate: PPL = 5.2000 +/- 0.1\nq4_k_m Final estimate: PPL = 5.4600 +/- 0.1\n")
    gm = gguf_quality_table(g)
    assert "| Q8_0 | 5.200 |  |" in gm and "| Q4_K_M | 5.460 | +5.0% |" in gm
