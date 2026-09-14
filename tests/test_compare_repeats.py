"""compare --repeats: rows measured round-robin, each standing for its median run, overlaps flagged as noise."""

from __future__ import annotations

import pytest

from polyserve.bench import compare, to_markdown
from polyserve.bench.compare import ComparisonResult, ComparisonRow, combine_runs, noise_notes
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config
from polyserve.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner


class DriftingRunner(CompareRunner):
    """The second run of each config is 10% faster and the third 10% slower, so the median is the first."""

    def __init__(self):
        super().__init__()
        self.stages, self.count = [], {}

    def run(self, cfg, stage):
        self.stages.append(stage)
        res = super().run(cfg, stage)
        n = self.count[cfg.key()] = self.count.get(cfg.key(), 0) + 1
        f = {1: 1.0, 2: 1.1, 3: 0.9}[n]
        if res.ok:
            for level in res.metrics.by_concurrency.values():
                level.tok_s *= f
            res.metrics.tok_s *= f
        return res


@pytest.mark.usefixtures("no_network")
def test_rows_are_measured_round_robin_and_stand_for_their_median_run(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    runner = DriftingRunner()
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"), runner=runner,
                     repeats=3)
    assert [s.split(":", 1)[1] for s in runner.stages] == ["polyserve", "vllm-default", "vllm-fp8-default"] * 3
    ps = result.polyserve_row
    assert ps.repeats == 3 and len(ps.runs_tok_s) == 3
    assert ps.scored_tok_s == sorted(ps.runs_tok_s)[1]  # the median run, not an average
    assert ps.tok_s_spread_pct == pytest.approx((max(ps.runs_tok_s) - min(ps.runs_tok_s)) / ps.scored_tok_s * 100)
    assert any("measured 3 times, interleaved" in n for n in result.notes)
    assert "median of 3" in to_markdown(result)


def _row(label, runs, ok=True):
    good = sorted(v for v in runs if v is not None)
    return ComparisonRow(label=label, runtime="vllm", config=Config(backend="vllm", quant="bf16"), config_key=label,
                         ok=ok, scored_tok_s=good[(len(good) - 1) // 2] if good else None, runs_tok_s=runs,
                         repeats=len(runs))


def test_a_failed_run_is_kept_and_the_median_comes_from_the_rest():
    ok = [_row("p", [v]) for v in (500.0, 520.0)]
    bad = ComparisonRow(label="p", runtime="vllm", config=Config(backend="vllm", quant="bf16"), config_key="p",
                        ok=False, error="OOM")
    row = combine_runs([ok[0], bad, ok[1]])
    assert row.runs_tok_s == [500.0, None, 520.0] and row.scored_tok_s == 500.0
    assert row.tok_s_spread_pct == pytest.approx(4.0)
    assert not combine_runs([bad, bad]).ok


def test_overlapping_runs_are_flagged_as_noise_and_separated_ones_are_not():
    res = ComparisonResult(polyserve_version="x", hardware_hash="h", gpu="A40", cpu="c", model_id="m",
                           objective="balanced", workload="chat", workload_spec={}, ttft_ceiling_ms=500,
                           rows=[_row("polyserve", [510.0, 530.0, 520.0]), _row("vllm-default", [470.0, 515.0, 480.0]),
                                 _row("llamacpp-cuda-default", [180.0, 190.0, 185.0])])
    notes = noise_notes(res)
    assert len(notes) == 1
    assert notes[0].startswith("vllm-default: its runs (470-515 tok/s) overlap PolyServe's (510-530)")
