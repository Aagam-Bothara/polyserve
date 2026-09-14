"""benchmarks/rescore_ttft.py: the same recorded rows, ranked by the median and by the 95th percentile."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "rescore_ttft.py"


def _load():
    spec = importlib.util.spec_from_file_location("rescore_ttft", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(label: str, levels) -> dict:
    """levels: {concurrency: (tok/s, TTFT p50, TTFT p95)}"""
    by = {str(c): {"tok_s": t, "ttft_ms": p50, "ttft_p95_ms": p95, "tpot_ms": 30.0, "requests": 16,
                   "output_tokens": 2048, "concurrency": c} for c, (t, p50, p95) in levels.items()}
    top = max(by.values(), key=lambda m: m["tok_s"])
    return {"label": label, "ok": True, "config": {"backend": "vllm", "quant": "fp8"}, "config_key": "vllm/fp8",
            "metrics": {**top, "by_concurrency": by}}


def test_the_p95_rule_serves_fewer_users(tmp_path, capsys):
    # The L4 on sharegpt: by the median the pick ran 32 users; its 95th percentile broke the ceiling there.
    data = {"objective": "balanced", "workload": "sharegpt", "gpu": "NVIDIA L4", "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "ttft_ceiling_ms": 1000.0, "tpot_ceiling_ms": 100.0,
            "rows": [_row("polyserve", {8: (212.0, 285.0, 429.0), 32: (738.0, 985.0, 1227.0)}),
                     _row("vllm-default", {8: (130.0, 300.0, 500.0), 32: (410.0, 628.0, 1664.0)})]}
    (tmp_path / "l4.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "profile.json").write_text(json.dumps({"calibration_table": []}), encoding="utf-8")  # skipped
    mod = _load()
    ours = mod.rescore(tmp_path / "l4.json")["rows"]["polyserve"]
    assert (ours[50]["concurrency"], ours[95]["concurrency"]) == (32, 8)
    mod.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "| p50 | 738 (at 32) |" in out and "410 (+80%)" in out
    assert "| p95 | 212 (at 8) |" in out and "130 (+63%)" in out
    assert "1 of 1 picks" in out
