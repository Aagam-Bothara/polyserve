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


def _probe(bf16_answers, awq_answers, bf16_repeat=None) -> QualityProbe:
    """A probe with both precisions already measured. `bf16_repeat` is the reference's second pass, which sets
    the noise floor; by default it is identical, so the floor is zero."""
    p = QualityProbe(prompts=["q1", "q2", "q3", "q4"])
    p.answers[("vllm", "bf16")] = bf16_answers
    p.answers[("vllm", "awq")] = awq_answers
    p.repeats[("vllm", "bf16")] = bf16_repeat if bf16_repeat is not None else list(bf16_answers)
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


def test_drift_is_reported_next_to_the_reference_noise():
    """Both numbers, because one is meaningless without the other: on a 14B the floor was 10% and every cheaper
    precision sat at 83-100%, which says the comparison saturates rather than that those weights are 8x worse."""
    p = _probe(["1", "2", "3", "4"], ["1", "2", "9", "8"], bf16_repeat=["1", "2", "3", "9"])

    assert p.drift("vllm", "awq") == 0.5
    assert p.noise("vllm") == 0.25  # identical weights, one answer different between two passes

    notes = p.summary()
    assert any("quality vllm/awq: 50% of answers differ from vllm/bf16" in n for n in notes)
    assert any("differs from itself on 25%" in n for n in notes)
    assert not any("allowed" in n or "over the" in n for n in notes)  # reported, never enforced


def test_a_precision_is_answered_twice(monkeypatch):
    """The second pass is what makes the first interpretable, so record() must take it."""
    import polyserve.quality as Q

    calls = []
    monkeypatch.setattr(Q, "greedy_answers", lambda url, prompts, *a, **kw: calls.append(url) or ["a"] * len(prompts))
    probe = Q.QualityProbe(prompts=["q1", "q2"])

    probe.record("http://engine", "vllm", "fp8")

    assert len(calls) == 2
    assert probe.answers[("vllm", "fp8")] and probe.repeats[("vllm", "fp8")]
    assert probe.noise("vllm") == 0.0


def test_nothing_is_refused_on_drift_but_it_is_recorded():
    """Drift gating was withdrawn: the comparison saturates, so the faster precision still wins and the profile
    simply carries what it drifted."""
    search = StagedSearch(objective="throughput", runner=FakeRunner(),
                          quality=_probe(["1", "2", "3", "4"], ["1", "9", "9", "9"]))

    winner, _ = search.run(FEASIBLE)

    assert winner is not None and winner.config.quant == "awq"  # it measured 200 tok/s and keeps its win
    assert search.refused == set()
    assert any("quality vllm/awq: 75% of answers differ" in n for n in search.notes)


def test_a_precision_on_the_refused_list_can_never_win():
    """Nothing populates `refused` today, but the invariant is what the drift gate got wrong: dropping a
    precision from stage 1's `kept` only stopped further tuning, and the argmax over every successful trial
    served it anyway. Any future gate adds here."""
    search = StagedSearch(objective="throughput", runner=FakeRunner())
    search.refused.add(("vllm", "awq"))

    winner, _ = search.run(FEASIBLE)

    assert winner is not None and winner.config.quant == "bf16"  # awq is faster and still cannot be chosen
    assert any(r.config.quant == "awq" and r.ok for r in search.results)  # still in the table


def test_a_probe_that_gets_no_answers_records_nothing(monkeypatch):
    """Every request failed. Empty answers compare equal to any other empty set, so recording them would report
    perfect agreement and wave the precision through."""
    import polyserve.quality as Q

    monkeypatch.setattr(Q, "greedy_answers", lambda url, prompts, *a, **kw: ["" for _ in prompts])
    probe = Q.QualityProbe(prompts=["q1", "q2"])

    probe.record("http://engine", "vllm", "fp8")

    assert probe.answers == {} and probe.repeats == {}
    assert probe.wanted("vllm", "fp8")  # so a later trial of the same precision tries again


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

    probe = quality_probe_for(SearchOptions(quality_probe=True), wl)

    assert probe is not None
    assert probe.prompts and len(probe.prompts) <= 48  # this file gives each level 20 of its 60 prompts
    assert wl.prompts == []  # the workload the trials measure is left untouched


def test_the_probe_gathers_prompts_across_levels_up_to_its_cap(tmp_path):
    """A workload hands out one level's prompts at a time — 32 on the chat preset — so reaching 48 means drawing
    from a second level. Enough that one changed answer is about 2% rather than the 6% sixteen prompts gave."""
    f = tmp_path / "many.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": f"question {i} about something"}) for i in range(600)),
                 encoding="utf-8")
    wl = workload_from_file(f, template=get_workload("chat"))
    assert wl.n_prompts == 32  # one level's worth, which is less than the cap

    probe = quality_probe_for(SearchOptions(quality_probe=True), wl)

    assert probe is not None and len(probe.prompts) == 48
    assert len(set(probe.prompts)) == 48  # drawn at different offsets, so no prompt is answered twice over


def test_no_probe_when_the_gate_was_not_asked_for():
    assert quality_probe_for(SearchOptions(), get_workload("chat")) is None


def _plan_with(quants, spec_id="org/big-model"):
    from polyserve.models import ModelSpec
    from polyserve.pipeline import PlanResult

    plan = PlanResult(hw=None, spec=ModelSpec(hf_id=spec_id), candidates=["vllm"])
    plan.feasible["vllm"] = [(Config(backend="vllm", quant=q, ctx=4096, batch=16), None) for q in quants]
    return plan


def test_a_card_that_cannot_hold_full_precision_says_so(caplog):
    """`--quant auto` skips 4-bit checkpoints on quality grounds, which is the right default and the wrong
    silence: on a 14B at 24 GB the skipped ones measured 3.9x faster for about 2 GSM8K points."""
    from polyserve.pipeline import _warn_if_only_quantized_fits

    with caplog.at_level("WARNING"):
        _warn_if_only_quantized_fits(_plan_with(["fp8"]), SearchOptions())

    assert "does not fit this GPU at full precision" in caplog.text
    assert "--quant auto,awq,gptq" in caplog.text


def test_no_such_warning_when_full_precision_fits_or_the_user_chose(caplog):
    from polyserve.pipeline import _warn_if_only_quantized_fits

    with caplog.at_level("WARNING"):
        _warn_if_only_quantized_fits(_plan_with(["bf16", "fp8"]), SearchOptions())  # bf16 fits: nothing to say
        _warn_if_only_quantized_fits(_plan_with(["fp8"]), SearchOptions(quants=["fp8"]))  # their own choice

    assert caplog.text == ""


def test_a_probe_records_each_precision_once():
    p = QualityProbe(prompts=["q1"])
    assert p.wanted("vllm", "bf16")
    p.answers[("vllm", "bf16")] = ["a"]
    assert not p.wanted("vllm", "bf16")
