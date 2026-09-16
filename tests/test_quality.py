from __future__ import annotations

import json

from polyserve.calibrate.search import StagedSearch
from polyserve.calibrate.workload import get_workload, workload_from_file
from polyserve.models import Config, TrialMetrics, TrialResult
from polyserve.pipeline import SearchOptions, quality_probe_for
from polyserve.quality import QualityProbe, agreement, precision_rank

FEASIBLE = [Config(backend="vllm", quant=q, ctx=4096, batch=b, gpu_memory_utilization=0.9)
            for q in ("bf16", "awq") for b in (16, 64)]


class FakeRunner:
    """Every trial succeeds; the 4-bit weights are faster, so only quality can keep them out."""

    def __init__(self):
        self.ran = []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.ran.append(cfg.key())
        m = TrialMetrics(tok_s=200.0 if cfg.quant == "awq" else 100.0, requests=32, output_tokens=2048,
                         concurrency=8, ttft_ms=50.0)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


def _probe(bf16_answers, awq_answers, tolerance=0.02) -> QualityProbe:
    p = QualityProbe(prompts=["q1", "q2", "q3", "q4"], tolerance=tolerance)
    p.answers[("vllm", "bf16")] = bf16_answers
    p.answers[("vllm", "awq")] = awq_answers
    return p


def test_agreement_ignores_whitespace_but_not_content():
    assert agreement(["a b", "c"], ["a  b\n", " c "]) == 1.0
    assert agreement(["a", "b", "c", "d"], ["a", "b", "c", "X"]) == 0.75
    assert agreement([], []) == 1.0


def test_the_reference_is_the_most_faithful_precision_that_ran():
    assert precision_rank("bf16") > precision_rank("fp8") > precision_rank("awq")
    p = _probe(["1", "2", "3", "4"], ["1", "2", "3", "4"])
    assert p.reference_for("vllm") == "bf16"
    assert p.drift("vllm", "bf16") is None  # the reference has nothing to differ from
    assert p.drift("vllm", "awq") == 0.0
    assert p.too_far("vllm", "awq") is None


def test_a_precision_that_answers_differently_is_refused():
    p = _probe(["1", "2", "3", "4"], ["1", "2", "9", "8"])  # half the answers changed
    assert p.drift("vllm", "awq") == 0.5
    why = p.too_far("vllm", "awq")
    assert why and "50%" in why and "--max-quality-loss" in why


def test_stage_one_drops_the_drifting_precision_even_though_it_is_faster():
    search = StagedSearch(objective="throughput", runner=FakeRunner(),
                          quality=_probe(["1", "2", "3", "4"], ["1", "9", "9", "9"]))
    kept = search.stage_quant(FEASIBLE)

    assert kept == [("vllm", "bf16")]  # awq measured 200 tok/s and still lost its place
    assert any("answered 75% of 4 prompts differently" in n for n in search.notes)
    assert any(n.startswith("quality reference vllm/bf16") for n in search.notes)


def test_without_a_probe_the_faster_precision_is_kept():
    search = StagedSearch(objective="throughput", runner=FakeRunner())
    assert search.stage_quant(FEASIBLE)[0] == ("vllm", "awq")


def test_a_file_workload_still_gets_a_probe(tmp_path):
    """Regression: a file workload carries no prompts until measurement materialises them per level, so a guard
    on `workload.prompts` disabled the gate on exactly the workloads a user would gate. It shipped inert."""
    f = tmp_path / "prompts.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": f"question {i} about something"}) for i in range(60)),
                 encoding="utf-8")
    wl = workload_from_file(f, template=get_workload("chat"))
    assert wl.prompts == []  # the condition that broke it

    probe = quality_probe_for(SearchOptions(max_quality_loss=0.02), wl)

    assert probe is not None and len(probe.prompts) == 16 and probe.tolerance == 0.02
    assert wl.prompts == []  # the workload the trials measure is left untouched


def test_no_probe_when_the_gate_was_not_asked_for():
    assert quality_probe_for(SearchOptions(), get_workload("chat")) is None


def test_a_probe_records_each_precision_once():
    p = QualityProbe(prompts=["q1"], tolerance=0.02)
    assert p.wanted("vllm", "bf16")
    p.answers[("vllm", "bf16")] = ["a"]
    assert not p.wanted("vllm", "bf16")
