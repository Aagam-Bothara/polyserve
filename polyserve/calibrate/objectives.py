"""The four objectives. All are constrained argmax; no weighted score formula in v1.

| objective   | rule                                   |
|-------------|----------------------------------------|
| throughput  | max tok/s                              |
| latency     | min TTFT   s.t. tok/s >= floor         |
| balanced    | max tok/s  s.t. TTFT   <= ceiling      |
| efficiency  | min J/tok  s.t. tok/s >= floor         |

Floors default to a fraction of the best observed tok/s; the ceiling defaults to an absolute
TTFT. If nothing satisfies the constraint, the least-violating candidate is chosen and the
result is flagged as relaxed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from polyserve.models import OBJECTIVES, TrialResult


@dataclass
class Constraints:
    ttft_ceiling_ms: float = 500.0  # balanced
    tok_s_floor_frac: float = 0.5  # latency / efficiency: floor = frac x best tok/s
    tok_s_floor_abs: Optional[float] = None  # overrides the fraction when set

    def tok_s_floor(self, results: Sequence[TrialResult]) -> float:
        if self.tok_s_floor_abs is not None:
            return self.tok_s_floor_abs
        best = max((r.metrics.tok_s for r in results), default=0.0)
        return self.tok_s_floor_frac * best


@dataclass
class Ranked:
    result: TrialResult
    score: float  # lower is better
    feasible: bool
    violation: float  # how far outside the constraint (0 if feasible)


def _ok_results(results: Sequence[TrialResult]) -> List[TrialResult]:
    return [r for r in results if r.ok]


def _ttft(r: TrialResult) -> float:
    t = r.metrics.ttft_ms
    return t if (t is not None and math.isfinite(t)) else math.inf


def _score_and_constraint(
    objective: str, cons: Constraints, results: Sequence[TrialResult]
) -> Tuple[Callable[[TrialResult], float], Callable[[TrialResult], float]]:
    """Return (score fn: lower better, violation fn: 0 when feasible)."""
    if objective == "throughput":
        return (lambda r: -r.metrics.tok_s), (lambda r: 0.0)
    if objective == "latency":
        floor = cons.tok_s_floor(results)
        return _ttft, (lambda r: max(0.0, floor - r.metrics.tok_s))
    if objective == "balanced":
        return (lambda r: -r.metrics.tok_s), (lambda r: max(0.0, _ttft(r) - cons.ttft_ceiling_ms))
    if objective == "efficiency":
        floor = cons.tok_s_floor(results)

        def score(r: TrialResult) -> float:
            j = r.metrics.joules_per_token
            if j is None or not math.isfinite(j):
                # No energy telemetry: fall back to ordering by tok/s so the objective still resolves.
                return 1e9 - r.metrics.tok_s
            return j

        return score, (lambda r: max(0.0, floor - r.metrics.tok_s))
    raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")


def rank(results: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None) -> List[Ranked]:
    """Best first. Feasible candidates always precede infeasible ones."""
    cons = cons or Constraints()
    ok = _ok_results(results)
    if not ok:
        return []
    score, violation = _score_and_constraint(objective, cons, ok)
    ranked = [Ranked(result=r, score=score(r), feasible=violation(r) == 0.0, violation=violation(r)) for r in ok]
    ranked.sort(key=lambda x: (not x.feasible, x.violation if not x.feasible else 0.0, x.score))
    return ranked


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
    if objective == "efficiency" and top.result.metrics.joules_per_token is None:
        notes.append("no energy telemetry available; efficiency objective fell back to tok/s ordering")
    return top.result, notes
