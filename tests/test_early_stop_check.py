"""benchmarks/early_stop_check.py: recorded trials re-scored as a top-down run would have measured them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "early_stop_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("early_stop_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trial(batch: int, levels) -> dict:
    """levels: {concurrency: (tok/s, TTFT p95 ms, seconds)}"""
    by = {str(c): {"tok_s": t, "ttft_ms": p95 / 2, "ttft_p95_ms": p95, "tpot_ms": 20.0, "requests": 8,
                   "output_tokens": 512, "concurrency": c, "duration_s": s} for c, (t, p95, s) in levels.items()}
    top = max(by.values(), key=lambda m: m["tok_s"])
    return {"config": {"backend": "vllm", "quant": "bf16", "batch": batch}, "stage": "batch",
            "metrics": {**top, "by_concurrency": by}}


def test_a_top_down_run_keeps_scores_and_skips_the_low_levels(tmp_path, capsys):
    profile = {"objective": "balanced", "calibration_seconds": 600.0,
               "notes": ["workload: chat: ..., TTFT ceiling 500 ms, TPOT ceiling 50 ms"],
               "calibration_table": [
                   _trial(64, {1: (60.0, 90.0, 30.0), 4: (220.0, 200.0, 20.0), 8: (400.0, 450.0, 20.0)}),
                   _trial(16, {1: (58.0, 95.0, 30.0), 4: (210.0, 300.0, 20.0), 8: (380.0, 900.0, 20.0)})]}
    (tmp_path / "p.json").write_text(json.dumps(profile), encoding="utf-8")
    mod = _load()
    r = mod.check(tmp_path / "p.json", 95)
    assert (r["trials"], r["changed"], r["same_pick"]) == (2, 0, True)
    assert r["skipped_s"] == 30.0 + 30.0 + 20.0  # batch 64 stops at 8 users; batch 16 breaks 500 ms there, stops at 4
    mod.main([str(tmp_path / "p.json")])
    assert "0 of 2 trial scores changed; 1 of 1 picks kept; 13% of calibration time skipped" in capsys.readouterr().out
