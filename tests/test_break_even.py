"""benchmarks/break_even.py: hours of serving before a calibration has paid for itself."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "break_even.py"


def _load():
    spec = importlib.util.spec_from_file_location("break_even", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(label, tok_s, meets_slo=True, ok=True, calibration_seconds=0.0):
    return {"label": label, "ok": ok, "scored_tok_s": tok_s, "meets_slo": meets_slo,
            "calibration_seconds": calibration_seconds}


def _write(tmp_path: Path, name: str, rows) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"gpu": "NVIDIA A40", "model_id": "Qwen/Qwen2.5-3B-Instruct", "workload": "dolly",
                             "rows": rows}), encoding="utf-8")
    return p


def test_the_dolly_pick_repays_its_calibration_in_hours(tmp_path, capsys):
    # The p95 re-run on an A40: 46 minutes of calibration, 619 tok/s against stock vLLM's 504 and SGLang's 475.
    mod = _load()
    p = _write(tmp_path, "a.json", [_row("polyserve", 619.0, calibration_seconds=2768.0),
                                    _row("vllm-default", 504.0), _row("sglang-default", 475.0)])
    r = mod.break_even(p)
    assert r["stock"] == "vllm-default" and abs(r["hours"] - 2768 * 504 / 115 / 3600) < 1e-9  # about 3.4 hours
    mod.main([str(tmp_path)])
    assert "| 46 min | 619 | vllm-default 504 | 3.4 h |" in capsys.readouterr().out


def test_a_pick_no_faster_than_stock_never_breaks_even(tmp_path):
    p = _write(tmp_path, "b.json", [_row("polyserve", 499.0, calibration_seconds=564.0), _row("vllm-default", 509.0)])
    assert _load().break_even(p)["hours"] == math.inf


def test_stock_rows_that_miss_the_limits_or_fail_do_not_count(tmp_path):
    mod = _load()
    p = _write(tmp_path, "c.json", [_row("polyserve", 195.0, calibration_seconds=3000.0),
                                    _row("vllm-default", 17.0, meets_slo=False), _row("vllm-fp8-default", None, ok=False)])
    r = mod.break_even(p)
    assert r["stock"] is None and mod._cell(r) == "no stock setup met the limits"
    assert mod.break_even(_write(tmp_path, "d.json", [_row("vllm-default", 504.0)])) is None  # no calibrated pick
