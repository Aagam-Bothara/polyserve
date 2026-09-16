"""How sure a measured percentile is: a distribution-free interval from the samples alone, and — just as
important — what confidence a given sample count can actually support.

A trial's p95 time to first token rests on a few dozen requests (32 at 8 users on the chat presets), so its
second-slowest request can decide whether a level meets a ceiling: on an L4 a pick measured 999 ms against a
1000 ms ceiling in calibration and was served at a lower load after re-measurement. For any percentile q the
true value lies between two order statistics of the samples, and which two follows from the binomial count of
samples that fall below it, with no assumption about the shape of the latency distribution.

The sample count caps what can be claimed, and the cap bites exactly where it matters. The largest of n samples
exceeds the true q-quantile with probability 1 - q**n, which for q = 0.95 and n = 32 is 80.6% — so the slowest
of 32 requests is **not** a 95% upper bound on p95, and calling it one would overstate the evidence. By default
an endpoint the samples cannot support is returned as infinity rather than silently clamped to the extreme
sample. A two-sided 95% interval for p95 needs 72 samples; a one-sided 95% upper bound needs 59.

`unbounded=False` restores the clamped form — the widest interval the order statistics can express — for callers
that want a screen rather than a guarantee. Those callers should report `attained_confidence(n, q)`, not 95%.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def _binomial_cdf(n: int, q: float) -> List[float]:
    """P(B <= i) for B ~ Binomial(n, q), i = 0..n: how many samples fall below the true quantile."""
    out: List[float] = []
    total = 0.0
    for i in range(n + 1):
        total += math.comb(n, i) * q ** i * (1 - q) ** (n - i)
        out.append(total)
    return out


def attained_confidence(n: int, q: float = 0.95, two_sided: bool = True) -> float:
    """The most confidence n samples can carry for the q-quantile, using the extreme order statistics.

    Two-sided, that is P(min <= quantile <= max) = 1 - q**n - (1-q)**n; one-sided upper, 1 - q**n. At q = 0.95
    and n = 32 the two-sided answer is 0.806, which is why a p95 from 32 requests is not a 95% claim.
    """
    if n <= 0:
        return 0.0
    if two_sided:
        return max(0.0, 1.0 - q ** n - (1.0 - q) ** n)
    return max(0.0, 1.0 - q ** n)


def samples_needed(q: float = 0.95, confidence: float = 0.95, two_sided: bool = True) -> int:
    """The smallest sample count at which `quantile_interval` returns finite endpoints at `confidence`.

    Stricter than `attained_confidence`, and deliberately so: that function reports the coverage of the widest
    interval the samples can express (1 - q**n - (1-q)**n), while the interval here splits its miss probability
    between the two tails, so each tail must fall under (1 - confidence)/2. For q = 0.95 at 95% confidence that
    is 72 samples two-sided and 59 for a one-sided upper bound.
    """
    tail = (1.0 - confidence) / 2 if two_sided else (1.0 - confidence)
    n = 1
    while n <= 100_000:  # far past any workload; stop rather than spin
        if q ** n <= tail and (not two_sided or (1.0 - q) ** n <= tail):
            return n
        n += 1
    return n


def quantile_interval(samples: Sequence[float], q: float, confidence: float = 0.95,
                      unbounded: bool = True) -> Tuple[float, float]:
    """(low, high): the q-quantile lies between them with at least `confidence` probability.

    An endpoint the sample count cannot support comes back as -inf or +inf, so the caller can see that the
    samples do not settle the question — with 32 samples and q = 0.95 the upper endpoint is +inf. Pass
    `unbounded=False` for the older clamped form, which uses the smallest and largest samples instead; that is
    a screen at `attained_confidence(len(samples), q)`, not at `confidence`.
    """
    x = sorted(samples)
    n = len(x)
    if n == 0:
        return math.nan, math.nan
    tail = (1.0 - confidence) / 2
    cdf = _binomial_cdf(n, q)
    lows = [r for r in range(1, n + 1) if cdf[r - 1] <= tail]
    highs = [r for r in range(1, n + 1) if cdf[r - 1] >= 1 - tail]
    if lows:
        low = x[max(lows) - 1]
    else:
        low = x[0] if not unbounded else -math.inf
    if highs:
        high = x[min(highs) - 1]
    else:
        high = x[-1] if not unbounded else math.inf
    return low, high
