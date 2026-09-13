"""The GSM8K grader in benchmarks/task_quality.py (no GPU or dataset needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "task_quality.py"
_spec = importlib.util.spec_from_file_location("task_quality", _PATH)
TQ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TQ)


def test_final_number_prefers_the_stated_answer():
    assert TQ.final_number("3 + 4 = 7, then 7 * 2 = 14.\nThe answer is 14.") == 14.0
    assert TQ.final_number("The answer is: $1,250") == 1250.0
    assert TQ.final_number("First 3 apples, then 5. The answer is 8. Check: 8 - 5 = 3") == 8.0
    assert TQ.final_number("So we get 42 in total") == 42.0  # no statement: the last number
    assert TQ.final_number("The answer is -3.5") == -3.5
    assert TQ.final_number("no numbers here") is None


def test_reference_and_interval():
    assert TQ.reference("Natalia sold 48/2 = 24 clips.\n#### 1,072") == 1072.0
    lo, hi = TQ.wilson(80, 100)
    assert lo == pytest.approx(0.711, abs=0.002) and hi == pytest.approx(0.867, abs=0.002)
    assert TQ.wilson(0, 0) == (0.0, 0.0)
