from __future__ import annotations

import json
from pathlib import Path

from polyserve.bench.compare import ComparisonResult, ComparisonRow
from polyserve.bench.report import (
    analyse,
    best_default,
    load_results,
    render_markdown,
    render_svg,
    summarize,
    write_report,
)
from polyserve.models import Config, TrialMetrics


def _row(label, runtime, tok_s, ttft, slo=500.0, ok=True, jpt=None, calib=(0, 0)) -> ComparisonRow:
    cfg = Config(backend=runtime, quant="bf16", ctx=4096, batch=64)
    r = ComparisonRow(label=label, runtime=runtime, config=cfg, config_key=cfg.key(), ok=ok,
                      error=None if ok else "boom", metrics=TrialMetrics(tok_s=tok_s, ttft_ms=ttft),
                      calibration_seconds=calib[0], calibration_trials=calib[1])
    if ok:
        r.scored_concurrency = 8
        r.scored_tok_s = tok_s
        r.scored_ttft_ms = ttft
        r.scored_joules_per_token = jpt
        r.meets_slo = ttft <= slo
    return r


def _result(gpu, model, workload, rows, ceiling=500.0) -> ComparisonResult:
    return ComparisonResult(polyserve_version="0.1.0", hardware_hash="abc", gpu=gpu, cpu="cpu", model_id=model,
                            objective="balanced", workload=workload, workload_spec={}, ttft_ceiling_ms=ceiling,
                            rows=rows)


def test_best_default_prefers_slo_meeting_rows():
    res = _result("RTX 3090", "q/3b", "chat", [
        _row("polyserve", "vllm", 1000, 40),
        _row("vllm-default", "vllm", 1200, 900),  # faster but misses the SLO
        _row("llamacpp-cuda-default", "llamacpp-cuda", 500, 100),
        _row("ollama-default", "ollama", 300, 80),
    ])
    base, met = best_default(res)
    assert base.label == "llamacpp-cuda-default" and met
    c = analyse(res)
    assert c.valid and round(c.tok_s_gain_pct) == 100 and c.ttft_delta_ms == -60


def test_best_default_falls_back_when_nothing_meets_slo():
    res = _result("A100", "q/8b", "rag", [
        _row("polyserve", "vllm", 800, 900, slo=1500),
        _row("vllm-default", "vllm", 700, 3000, slo=1500),
        _row("sglang-default", "sglang", 750, 2500, slo=1500),
    ], ceiling=1500)
    base, met = best_default(res)
    assert base.label == "sglang-default" and not met
    s = summarize([res])
    assert s.scored and not s.like_for_like  # excluded from the headline
    assert "No combination" in s.headline()


def test_summary_headline_and_markdown():
    results = [
        _result("A100", "m/a", "chat", [_row("polyserve", "vllm", 1500, 50, jpt=0.5, calib=(600, 10)),
                                        _row("vllm-default", "vllm", 1000, 60, jpt=0.8)]),
        _result("A100", "m/a", "rag", [_row("polyserve", "vllm", 400, 700, slo=1500),
                                       _row("vllm-default", "vllm", 500, 800, slo=1500)], ceiling=1500),
        _result("3090", "m/b", "chat", [_row("polyserve", "llamacpp-cuda", 640, 72),
                                        _row("llamacpp-cuda-default", "llamacpp-cuda", 545, 952),
                                        _row("ollama-default", "ollama", 400, 300)]),
        _result("3090", "m/b", "generation", [_row("polyserve", "vllm", 0, 0, ok=False),
                                              _row("vllm-default", "vllm", 100, 50)]),
    ]
    s = summarize(results)
    assert len(s.like_for_like) == 3
    gains = sorted(round(c.tok_s_gain_pct) for c in s.like_for_like)
    assert gains == [-20, 50, 60]
    assert s.median_gain() == 50.0 and s.wins() == (2, 3)
    head = s.headline()
    assert "3 machine/model/workload combinations" in head and "+50%" in head and "winning 2 of 3" in head
    md = render_markdown(s)
    assert "| A100 | m/a | chat |" in md and "PolyServe failed" in md and "+50%" in md
    assert "600s / 10" in md and "+38%" in md  # energy gain (0.8 -> 0.5)


def test_svg_has_one_panel_per_combo_and_marks_slo():
    results = [
        _result("A100", "m/a", "chat", [_row("polyserve", "vllm", 1500, 50), _row("vllm-default", "vllm", 1000, 60)]),
        _result("3090", "m/b", "rag", [_row("polyserve", "vllm", 400, 700, slo=1500),
                                       _row("ollama-default", "ollama", 200, 900, slo=1500)], ceiling=1500),
    ]
    svg = render_svg(results)
    assert svg.startswith("<svg") and svg.count("PolyServe</text>") == 3  # one label per panel + the legend
    assert "SLO 500 ms" in svg and "SLO 1500 ms" in svg
    assert svg.count('class="slo"') == 2 and "ollama" in svg
    assert "no results yet" in render_svg([])


def test_write_report_roundtrip(tmp_path: Path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    res = _result("A100", "m/a", "chat", [_row("polyserve", "vllm", 1500, 50), _row("vllm-default", "vllm", 1000, 60)])
    (rdir / "abc__m__a__chat__balanced.json").write_text(res.model_dump_json(), encoding="utf-8")
    assert len(load_results(rdir)) == 1
    md, svg, summary = write_report(rdir)
    assert md.exists() and svg.exists() and md.parent == tmp_path
    assert "+50%" in md.read_text(encoding="utf-8")
    assert json.loads((rdir / "abc__m__a__chat__balanced.json").read_text())["rows"][0]["label"] == "polyserve"
    assert load_results(tmp_path / "missing") == []
