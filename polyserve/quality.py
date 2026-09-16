"""Does a cheaper precision still give the same answers?

Calibration measures speed. Quantization can also change what the model says, and a faster configuration that
answers differently is not obviously a better one. This module measures that directly, on the prompts the user
is already calibrating with, and without any labelled data: run the prompts greedily at the highest precision
that fits, run them again at each cheaper precision, and count how often the answers differ.

The comparison is deliberately blunt. Greedy decoding makes a run reproducible, so a difference is the
precision's doing rather than sampling noise, and the score is the share of prompts whose answers match after
whitespace is normalised. It says "these weights answer differently", not "these weights are worse": judging
better or worse needs labels or a judge, which `benchmarks/task_quality.py` does separately against GSM8K.

`--max-quality-loss` turns the measurement into a constraint: a precision whose answers drift further than the
tolerance is dropped from the search the same way the planner drops a configuration that cannot fit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

# Higher is more faithful: the reference is the most precise weight format that fits on the card.
PRECISION_RANK: Dict[str, int] = {
    "fp32": 100, "bf16": 90, "fp16": 90, "Q8_0": 70, "fp8": 60, "w8a8": 55,
    "Q6_K": 45, "Q5_K_M": 40, "awq": 30, "gptq": 30, "Q4_K_M": 25,
}


def precision_rank(quant: str) -> int:
    """Where a weight format sits from most to least faithful; unknown formats rank below every known one."""
    return PRECISION_RANK.get(quant, 0)


def normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").strip())


def agreement(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """The share of prompts whose answers match after whitespace normalisation, over the pairs that exist."""
    pairs = list(zip(reference, candidate))
    if not pairs:
        return 1.0
    same = sum(1 for a, b in pairs if normalise(a) == normalise(b))
    return same / len(pairs)


def greedy_answers(base_url: str, prompts: Sequence[str], max_tokens: int = 128,
                   timeout: float = 120.0) -> List[str]:
    """Answer each prompt once, greedily, against a running engine. Blank for any request that fails."""
    out: List[str] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for prompt in prompts:
            body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
            try:
                resp = client.post("/v1/completions", json=body)
                if resp.status_code != 200:
                    logger.debug("quality probe: HTTP %s", resp.status_code)
                    out.append("")
                    continue
                choices = resp.json().get("choices") or [{}]
                out.append(choices[0].get("text") or "")
            except (httpx.HTTPError, ValueError) as exc:
                logger.debug("quality probe failed: %s", exc)
                out.append("")
    return out


@dataclass
class QualityProbe:
    """Records each precision's greedy answers during calibration and says which ones drifted too far.

    The probe runs once per (backend, quant), on a server the search has already launched, so it costs one
    short generation pass per precision rather than anything per trial.
    """

    prompts: List[str]
    max_tokens: int = 128
    tolerance: float = 0.02  # --max-quality-loss: allowed share of answers that may differ from the reference
    timeout: float = 120.0
    answers: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def wanted(self, backend: str, quant: str) -> bool:
        return bool(self.prompts) and (backend, quant) not in self.answers

    def record(self, base_url: str, backend: str, quant: str) -> None:
        """Answer the prompts on this running engine, unless this precision has already been measured."""
        if not self.wanted(backend, quant):
            return
        self.answers[(backend, quant)] = greedy_answers(base_url, self.prompts, self.max_tokens, self.timeout)

    def reference_for(self, backend: str) -> Optional[str]:
        """The most faithful precision measured for this backend: what the others are compared against."""
        quants = [q for (b, q) in self.answers if b == backend]
        return max(quants, key=precision_rank) if quants else None

    def drift(self, backend: str, quant: str) -> Optional[float]:
        """Share of answers that differ from the reference precision, or None when there is nothing to compare."""
        ref = self.reference_for(backend)
        if ref is None or ref == quant:
            return None
        mine = self.answers.get((backend, quant))
        theirs = self.answers.get((backend, ref))
        if not mine or not theirs:
            return None
        return 1.0 - agreement(theirs, mine)

    def too_far(self, backend: str, quant: str) -> Optional[str]:
        """Why this precision should be dropped, or None to keep it."""
        d = self.drift(backend, quant)
        if d is None or d <= self.tolerance:
            return None
        ref = self.reference_for(backend)
        return (f"{backend}/{quant} answered {d:.0%} of {len(self.prompts)} prompts differently from "
                f"{backend}/{ref}, over the {self.tolerance:.0%} allowed by --max-quality-loss")

    def summary(self) -> List[str]:
        """One note per measured precision, for the profile: what drifted and what did not."""
        lines: List[str] = []
        for (backend, quant) in sorted(self.answers):
            d = self.drift(backend, quant)
            if d is None:
                lines.append(f"quality reference {backend}/{quant}: {len(self.prompts)} prompts answered greedily")
            else:
                lines.append(f"quality {backend}/{quant}: {d:.0%} of answers differ from "
                             f"{backend}/{self.reference_for(backend)}")
        return lines
