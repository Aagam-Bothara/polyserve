"""How sure a measured percentile is: a confidence interval from the samples alone.

A trial's p95 time to first token rests on a few dozen requests (32 at 8 users on the chat presets), so its
second-slowest request can decide whether a level meets a ceiling: on an L4 a pick measured 999 ms against a
1000 ms ceiling in calibration and was served at a lower load after re-measurement. For any percentile q the
true value lies between two order statistics of the samples, and which two follows from the binomial count of
samples that fall below it, with no assumption about the shape of the latency distribution.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def quantile_interval(samples: Sequence[float], q: float, confidence: float = 0.95) -> Tuple[float, float]:
    """(low, high): the q-quantile lies between them with at least `confidence` probability, or as close to that
    as the sample size allows. With 32 samples and q = 0.95 the upper end is simply the slowest sample."""
    x = sorted(samples)
    n = len(x)
    if n == 0:
        return math.nan, math.nan
    tail = (1.0 - confidence) / 2
    cdf, total = [], 0.0
    for i in range(n + 1):  # P(B <= i) for B ~ Binomial(n, q): how many samples fall below the true quantile
        total += math.comb(n, i) * q ** i * (1 - q) ** (n - i)
        cdf.append(total)
    low = max((r for r in range(1, n + 1) if cdf[r - 1] <= tail), default=1)
    high = min((r for r in range(1, n + 1) if cdf[r - 1] >= 1 - tail), default=n)
    return x[low - 1], x[high - 1]
