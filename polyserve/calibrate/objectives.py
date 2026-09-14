"""The four objectives. All are constrained argmax; no weighted score formula in v1.

| objective   | rule                                   |
|-------------|----------------------------------------|
| throughput  | max tok/s                              |
| latency     | min request latency (TTFT + TPOT x tokens) s.t. tok/s >= floor |
| balanced    | max tok/s  s.t. TTFT p95 <= ceiling    |
| efficiency  | min J/tok  s.t. tok/s >= floor         |

Floors default to a fraction of the best observed tok/s; the ceiling defaults to an absolute
TTFT. If nothing satisfies the constraint, the least-violating candidate is chosen and the
result is flagged as relaxed.

Each trial is measured at several concurrency levels. A trial is scored at whichever of its
levels best satisfies the objective (e.g. for `balanced`, the highest-throughput level whose
TTFT is still under the ceiling), so the winner is a (config, load) pair the server can run at.

Scores within `noise_tolerance` of the best are treated as ties and broken in favour of the
more capable configuration (larger context, then larger batch), since a 1% tok/s edge is
measurement noise and a bigger context window is a real capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from polyserve.models import OBJECTIVES, TrialMetrics, TrialResult


@dataclass
class Constraints:
    ttft_ceiling_ms: float = 500.0  # balanced
    # Which time to first token the ceiling applies to: 95 = the 95th percentile (1 request in 20 is
    # slower), 50 = the median. In 8 of 20 recorded comparisons (an A40, an L4, a CPU) the load a
    # pick was served at by the median broke the ceiling at p95, so the tail is the default.
    ttft_percentile: int = 95
    tok_s_floor_frac: float = 0.5  # latency / efficiency: floor = frac x best tok/s
    tok_s_floor_abs: Optional[float] = None  # overrides the fraction when set
    noise_tolerance: float = 0.02  # scores within 2% are ties -> prefer larger ctx, then batch
    # Per-token decode latency ceiling. Applied under every objective when set: TTFT constrains the
    # prefill phase, this constrains the decode phase.
    tpot_ceiling_ms: Optional[float] = None
    # A power-capped or clock-locked variant of the leading config may cost up to this much of its
    # score and still be chosen, when it uses less energy. The default keeps savings "free":
    # within run-to-run noise of the uncapped result.
    power_max_loss: float = 0.02

    def tok_s_floor(self, results: Sequence[TrialResult]) -> float:
        if self.tok_s_floor_abs is not None:
            return self.tok_s_floor_abs
        best = max((r.metrics.tok_s for r in results), default=0.0)
        return self.tok_s_floor_frac * best


@dataclass
class Ranked:
    result: TrialResult
    metrics: TrialMetrics  # the concurrency level the trial is scored at
    score: float  # lower is better
    feasible: bool
    violation: float  # how far outside the constraint (0 if feasible)

    @property
    def concurrency(self) -> int:
        return self.metrics.concurrency


def _ok_results(results: Sequence[TrialResult]) -> List[TrialResult]:
    return [r for r in results if r.ok]


def _ttft(m: TrialMetrics) -> float:
    t = m.ttft_ms
    return t if (t is not None and math.isfinite(t)) else math.inf


def _ttft_for(m: TrialMetrics, percentile: int) -> float:
    """The time to first token a ceiling is judged on: the 95th percentile when asked for and recorded."""
    if percentile >= 95 and m.ttft_p95_ms is not None and math.isfinite(m.ttft_p95_ms):
        return m.ttft_p95_ms
    return _ttft(m)


def _levels(r: TrialResult) -> List[TrialMetrics]:
    levels = [m for m in r.metrics.by_concurrency.values() if m.ok]
    return levels or [r.metrics]


def _e2e_latency(m: TrialMetrics) -> float:
    """Median request latency: time to first token plus the rest of the answer at TPOT.

    Counting decode makes the latency objective see speculative decoding, which leaves TTFT
    alone and cuts time per token.
    """
    ttft = _ttft(m)
    if not math.isfinite(ttft):
        return ttft
    per_request = (m.output_tokens / m.requests) if m.requests else 0.0
    tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
    return ttft + tpot * max(0.0, per_request - 1)


def _score_and_constraint(
    objective: str, cons: Constraints, results: Sequence[TrialResult]
) -> Tuple[Callable[[TrialMetrics], float], Callable[[TrialMetrics], float]]:
    """Return (score fn: lower better, violation fn: 0 when feasible), both over TrialMetrics."""
    if objective == "throughput":
        return (lambda m: -m.tok_s), (lambda m: 0.0)
    if objective == "latency":
        floor = cons.tok_s_floor(results)
        return _e2e_latency, (lambda m: max(0.0, floor - m.tok_s))
    if objective == "balanced":
        return (lambda m: -m.tok_s), (lambda m: max(0.0, _ttft_for(m, cons.ttft_percentile) - cons.ttft_ceiling_ms))
    if objective == "efficiency":
        floor = cons.tok_s_floor(results)

        def score(m: TrialMetrics) -> float:
            j = m.joules_per_token
            if j is None or not math.isfinite(j):
                # No energy telemetry: fall back to ordering by tok/s so the objective still resolves.
                return 1e9 - m.tok_s
            return j

        return score, (lambda m: max(0.0, floor - m.tok_s))
    raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")


def _sort_key(x: Ranked) -> Tuple[bool, float, float]:
    return (not x.feasible, x.violation if not x.feasible else 0.0, x.score)


def _capability(r: TrialResult) -> Tuple[int, int, float]:
    c = r.config
    mem = c.gpu_memory_utilization if c.gpu_memory_utilization is not None else float(c.n_gpu_layers or 0)
    return (c.ctx, c.batch, mem)


def _variant_key(r: TrialResult) -> str:
    """What makes two trials 'the same configuration at different power': everything but power."""
    d = getattr(r, "disagg", None)
    reps = getattr(r, "replicas", 1)
    return (r.config.base_key() + (f"|pd:{d.prefill.key()}" if d is not None else "")
            + (f"|x{reps}" if reps > 1 else ""))


def _strategies(r: TrialResult) -> int:
    """Optional strategies a config switches on beyond the engine's defaults.

    Within the noise band the config with fewer of them wins, so a strategy is adopted only when
    it measurably helps. A quantized KV cache or speculative decoding that merely ties is complexity
    (and, for the cache, a quality risk) with nothing to show for it.
    """
    c = r.config
    return sum((c.kv_dtype != "auto", c.spec_decode is not None, c.prefill_budget is not None,
                bool({"cache_reuse", "kv_unified"} & set(c.extra))))


def _joules(x: Ranked) -> float:
    j = x.metrics.joules_per_token
    return j if (j is not None and math.isfinite(j)) else math.inf


def _break_ties(ranked: List[Ranked], cons: Constraints) -> List[Ranked]:
    """Within the leading cluster of feasible, near-equal scores: fewer optional strategies first, then
    the larger config, then less energy.

    Two things join the leader's cluster: anything within `noise_tolerance` of its score, and any
    power-capped or clock-locked variant of the same configuration within `power_max_loss`.
    Variants share context, batch and memory settings, so the energy key decides between them,
    and a cap that saves joules at no measurable throughput cost wins over running uncapped.
    """
    if not ranked or not ranked[0].feasible:
        return ranked
    lead = ranked[0]
    span = abs(lead.score) * max(cons.noise_tolerance, 0.0)
    power_span = abs(lead.score) * max(cons.power_max_loss, cons.noise_tolerance, 0.0)
    lead_base = _variant_key(lead.result)

    def joins(x: Ranked) -> bool:
        if not x.feasible:
            return False
        gap = abs(x.score - lead.score)
        return gap <= span or (_variant_key(x.result) == lead_base and gap <= power_span)

    cluster = [x for x in ranked if joins(x)]
    if len(cluster) < 2:
        return ranked
    cluster.sort(key=lambda x: (_strategies(x.result), tuple(-v for v in _capability(x.result)), _joules(x)))
    rest = [x for x in ranked if x not in cluster]
    return cluster + rest


def rank(results: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None) -> List[Ranked]:
    """Best first. Feasible candidates always precede infeasible ones."""
    cons = cons or Constraints()
    ok = _ok_results(results)
    if not ok:
        return []
    score, violation = _score_and_constraint(objective, cons, ok)
    if cons.tpot_ceiling_ms is not None:
        phase_violation, ceiling = violation, cons.tpot_ceiling_ms

        def violation(m: TrialMetrics) -> float:  # noqa: F811  (the decode-phase SLO on top)
            tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
            return phase_violation(m) + max(0.0, tpot - ceiling)
    ranked: List[Ranked] = []
    for r in ok:
        # Score the trial at its best concurrency level for this objective.
        per_level = [
            Ranked(result=r, metrics=m, score=score(m), feasible=violation(m) == 0.0, violation=violation(m))
            for m in _levels(r)
        ]
        per_level.sort(key=_sort_key)
        ranked.append(per_level[0])
    ranked.sort(key=_sort_key)
    return _break_ties(ranked, cons)


def pick(
    results: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None
) -> Tuple[Optional[TrialResult], List[str]]:
    """Return (winner, notes). Winner is None only if no trial succeeded."""
    ranked = rank(results, objective, cons)
    if not ranked:
        return None, ["no successful trials"]
    top = ranked[0]
    notes: List[str] = []
    if not top.feasible:
        notes.append(
            f"no config satisfied the {objective} constraint; picked the least-violating one "
            f"(violation {top.violation:.1f})"
        )
    if objective == "efficiency" and top.metrics.joules_per_token is None:
        notes.append("no energy telemetry available; efficiency objective fell back to tok/s ordering")
    m = top.metrics
    notes.append(
        f"winner scored at concurrency {m.concurrency}: {m.tok_s:.1f} tok/s, "
        f"TTFT {m.ttft_ms:.0f} ms" + (f" (p95 {m.ttft_p95_ms:.0f})" if m.ttft_p95_ms is not None else "")
        + (f", {m.joules_per_token:.3f} J/tok" if m.joules_per_token else "")
    )
    return top.result, notes
