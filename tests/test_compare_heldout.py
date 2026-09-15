"""compare --eval-workload-file: the pick is calibrated on one workload and every row measured on another."""

from __future__ import annotations

import pytest

from polyserve.bench import compare
from polyserve.calibrate.workload import get_workload
from polyserve.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner


@pytest.mark.usefixtures("no_network")
def test_rows_measured_on_held_out_prompts_say_what_the_pick_was_calibrated_on(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    assert profile.workload == "default"
    held_out = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("chat"),
                       runner=CompareRunner())
    assert held_out.workload == "chat" and held_out.calibrated_on == "default"
    assert any("calibrated on `default`" in n and "never saw" in n for n in held_out.notes)
    same = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                   runner=CompareRunner())
    assert same.calibrated_on is None and not any("never saw" in n for n in same.notes)
