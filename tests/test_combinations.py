"""Stage 3f: strategies that are promising on their own are also measured in combination."""

from __future__ import annotations

from typing import Callable, Dict, List

from polyserve.calibrate.search import StagedSearch
from polyserve.models import Config, TrialMetrics, TrialResult

BASE = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=8192, batch=8, n_gpu_layers=37, n_batch=512)


class ScoredRunner:
    """Throughput from the strategies a config carries; records what it ran."""

    def __init__(self, score: Callable[[Config], float]):
        self.score = score
        self.ran: List[Config] = []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.ran.append(cfg)
        tok = self.score(cfg)
        m = TrialMetrics(tok_s=tok, ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048, concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))

    def keys(self, stage_cfgs=None) -> List[str]:
        return [c.key() for c in self.ran]


def measured_on_a40(cfg: Config) -> float:
    """llama.cpp, chat-system, A40: --kv-unified alone tied, speculation alone won, both won more."""
    unified, spec = bool(cfg.extra.get("kv_unified")), cfg.spec_decode is not None
    return {(False, False): 400.0, (True, False): 401.0, (False, True): 427.0, (True, True): 482.0}[(unified, spec)]


def _two_stages():
    def prefix(c: Config) -> List[Config]:
        return [] if c.extra.get("kv_unified") else [c.model_copy(update={"extra": {**c.extra, "kv_unified": True}})]

    def spec(c: Config) -> List[Config]:
        return [] if c.spec_decode else [c.model_copy(update={"spec_decode": "ngram:64"})]

    return [("prefix", prefix), ("spec", spec)]


def test_the_combination_one_dimension_at_a_time_missed_is_found():
    runner = ScoredRunner(measured_on_a40)
    winner, _ = StagedSearch(objective="throughput", runner=runner, variant_stages=_two_stages()).run([BASE])
    assert winner.config.spec_decode == "ngram:64" and winner.config.extra.get("kv_unified") is True
    assert winner.metrics.tok_s == 482.0 and winner.stage == "combine"
    off, _ = StagedSearch(objective="throughput", runner=ScoredRunner(measured_on_a40),
                          variant_stages=_two_stages(), combine=False).run([BASE])
    assert off.metrics.tok_s == 427.0  # one dimension at a time stops at speculation alone


def test_changes_that_clearly_lost_are_not_combined():
    def unified_hurts(cfg: Config) -> float:  # --kv-unified 20% worse: not a candidate
        return measured_on_a40(cfg) * (0.8 if cfg.extra.get("kv_unified") else 1.0)

    runner = ScoredRunner(unified_hurts)
    StagedSearch(objective="throughput", runner=runner, variant_stages=_two_stages()).run([BASE])
    assert len(runner.ran) == 3  # base, +kv_unified, +spec; nothing to combine


def test_a_tie_is_still_combined_but_only_adopted_if_it_wins():
    runner = ScoredRunner(lambda cfg: 400.0)  # everything ties
    winner, _ = StagedSearch(objective="throughput", runner=runner, variant_stages=_two_stages()).run([BASE])
    assert len(runner.ran) == 4  # the pair is measured once
    assert winner.config == BASE  # and the tie-break keeps the config with fewer strategies


BONUS: Dict[str, float] = {"kv_unified": 8, "cache_reuse": 4, "ngram:64": 20, "draft:m:16": 12, "q8_0": 2, "q4_0": -8}


def _bonus_score(cfg: Config) -> float:
    marks = [k for k in cfg.extra if cfg.extra[k]] + [cfg.spec_decode or "", cfg.kv_dtype]
    return 400.0 + sum(BONUS.get(m, 0.0) for m in marks)


def _three_stages():
    def kv(c: Config) -> List[Config]:
        return [c.model_copy(update={"kv_dtype": d}) for d in ("q8_0", "q4_0") if c.kv_dtype != d]

    def prefix(c: Config) -> List[Config]:
        return [c.model_copy(update={"extra": {**c.extra, k: True}}) for k in ("kv_unified", "cache_reuse")
                if k not in c.extra]

    def spec(c: Config) -> List[Config]:
        return [c.model_copy(update={"spec_decode": s}) for s in ("ngram:64", "draft:m:16") if c.spec_decode != s]

    return [("kv", kv), ("prefix", prefix), ("spec", spec)]


def test_combinations_are_bounded_and_the_best_estimate_goes_first():
    runner = ScoredRunner(_bonus_score)
    search = StagedSearch(objective="throughput", runner=runner, variant_stages=_three_stages(), max_combinations=3)
    winner, _ = search.run([BASE])
    combined = [r.config for r in search.results if r.stage == "combine"]
    assert len(combined) == 3
    first = combined[0]  # +q8_0 (+0.5%), +kv_unified (+2%), +ngram (+5%): the highest estimated sum
    assert first.kv_dtype == "q8_0" and first.extra == {"kv_unified": True} and first.spec_decode == "ngram:64"
    # It scores 430 against 428 without q8_0: inside the 2% noise band, so the tie-break drops the
    # strategy that added only noise and keeps the pair.
    assert winner.config.kv_dtype == "auto" and winner.config.extra == {"kv_unified": True}
    assert winner.config.spec_decode == "ngram:64" and winner.metrics.tok_s == 428.0


def test_a_change_adopted_first_is_dropped_when_a_later_one_does_better_without_it():
    """A40, extract: the fp8 cache won alone, the draft model then won on top of it, but the draft
    model without the fp8 cache was faster still. Single changes are re-measured on the base."""
    def extract(cfg: Config) -> float:
        fp8, draft = cfg.kv_dtype == "q8_0", cfg.spec_decode is not None
        return {(False, False): 442.0, (True, False): 526.0, (True, True): 610.0, (False, True): 728.0}[(fp8, draft)]

    def kv(c: Config) -> List[Config]:
        return [] if c.kv_dtype == "q8_0" else [c.model_copy(update={"kv_dtype": "q8_0"})]

    def spec(c: Config) -> List[Config]:
        return [] if c.spec_decode else [c.model_copy(update={"spec_decode": "draft:m:4"})]

    winner, _ = StagedSearch(objective="throughput", runner=ScoredRunner(extract),
                             variant_stages=[("kv", kv), ("spec", spec)]).run([BASE])
    assert winner.config.spec_decode == "draft:m:4" and winner.config.kv_dtype == "auto"
    assert winner.metrics.tok_s == 728.0 and winner.stage == "combine"


def test_undoing_an_adopted_change_is_tried_before_adding_more():
    """The same extract numbers, a prefill change worth 1%, and a budget of one combination. Summed
    gains would spend it on prefill + fp8 cache + draft model; undoing the fp8 cache goes first."""
    def extract(cfg: Config) -> float:
        fp8, draft = cfg.kv_dtype == "q8_0", cfg.spec_decode is not None
        tok = {(False, False): 442.0, (True, False): 526.0, (True, True): 610.0, (False, True): 728.0}[(fp8, draft)]
        return tok * (1.01 if cfg.prefill_budget == 8192 else 1.0)

    def prefill(c: Config) -> List[Config]:
        return [] if c.prefill_budget else [c.model_copy(update={"prefill_budget": 8192})]

    def kv(c: Config) -> List[Config]:
        return [] if c.kv_dtype == "q8_0" else [c.model_copy(update={"kv_dtype": "q8_0"})]

    def spec(c: Config) -> List[Config]:
        return [] if c.spec_decode else [c.model_copy(update={"spec_decode": "draft:m:4"})]

    search = StagedSearch(objective="throughput", runner=ScoredRunner(extract), max_combinations=1,
                          variant_stages=[("prefill", prefill), ("kv", kv), ("spec", spec)])
    winner, _ = search.run([BASE])
    (combined,) = [r.config for r in search.results if r.stage == "combine"]
    assert combined.kv_dtype == "auto" and combined.spec_decode == "draft:m:4"
    assert winner.config == combined and winner.metrics.tok_s == 728.0


def test_infeasible_combinations_are_skipped():
    runner = ScoredRunner(measured_on_a40)
    search = StagedSearch(objective="throughput", runner=runner, variant_stages=_two_stages(),
                          feasible_fn=lambda c: not (c.extra.get("kv_unified") and c.spec_decode))
    winner, _ = search.run([BASE])
    assert not [r for r in search.results if r.stage == "combine"] and winner.metrics.tok_s == 427.0
