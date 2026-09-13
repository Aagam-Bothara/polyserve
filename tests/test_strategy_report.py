"""The markdown report built from compare, ablation and quality outputs."""

from __future__ import annotations

import json

from polyserve.bench.ablation import (
    ablation_report,
    compare_table,
    gguf_quality_table,
    mcnemar_p,
    quality_table,
    task_quality_table,
)


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


def test_compare_table_labels_reruns_and_skips_other_json(tmp_path):
    rerun = tmp_path / "results-real" / "v3"
    rerun.mkdir(parents=True)
    f = rerun / "h__Qwen__Qwen2.5-3B-Instruct__extract__balanced.json"
    f.write_text(json.dumps({"model_id": "Qwen/Qwen2.5-3B-Instruct", "workload": "extract", "gpu": "NVIDIA A40",
                             "rows": [_row("polyserve", "vllm/bf16/b256", 731.0), _row("vllm-default", "k", 439.0)]}))
    profile = rerun / "extract-run2.profile.json"
    profile.write_text(json.dumps({"calibration_table": []}))
    md = compare_table([f, profile])
    assert "| A40 | Qwen2.5-3B-Instruct | extract (v3) | `vllm/bf16/b256` | 731 |" in md and "439 (+67%)" in md
    assert md.count("\n| ") == 1  # one data row: the profile is skipped


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


def test_task_quality_table_pairs_against_the_reference(tmp_path):
    t = tmp_path / "task-quality-3b.json"
    t.write_text(json.dumps({"model": "Qwen/Qwen2.5-3B-Instruct", "accuracy": {
        "bf16": {"correct": 1150, "n": 1319, "accuracy": 1150 / 1319, "ci95": [0.853, 0.889]},
        "fp8": {"correct": 1119, "n": 1319, "accuracy": 1119 / 1319, "ci95": [0.828, 0.867],
                "vs_bf16": {"lost": 65, "gained": 34}},
        "awq": {"error": "RuntimeError: engine failed"}}}))
    md = task_quality_table([t])
    assert "| Qwen2.5-3B-Instruct | bf16 | 87.2% | 85.3%–88.9% | reference | | |" in md
    assert "| fp8 | 84.8% | 82.8%–86.7% | -2.4 pts vs bf16 | 65 / 34 | 0.002 |" in md
    assert "| awq | failed |" in md


def test_mcnemar_exact_p_value():
    assert mcnemar_p(0, 0) == 1.0 and mcnemar_p(10, 10) == 1.0
    assert abs(mcnemar_p(65, 34) - 0.0024) < 0.0002  # GSM8K 3B, fp8 against bf16
    assert mcnemar_p(119, 53) < 0.001 and mcnemar_p(23, 20) > 0.7
