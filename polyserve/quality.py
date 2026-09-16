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


def _answered(answers: Sequence[str]) -> bool:
    """Whether a pass produced anything at all: all-empty means every request failed, and an empty set compares
    equal to any other empty set, which would read as perfect agreement."""
    return any(normalise(a) for a in answers)


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

    The probe runs once per (backend, quant), on a server the search has already launched, so it costs two
    short generation passes per precision rather than anything per trial.

    Two things keep it from refusing everything. It compares only the opening of each answer, because greedy
    decoding diverges once a single token differs and a 128-token completion will therefore disagree for almost
    any quantization; the first ~32 tokens say whether the model started the same answer. And every precision is
    answered twice, so the reference's disagreement with itself measures the floor — batched vLLM is not
    bit-reproducible — and a candidate must exceed that floor by the tolerance before it is refused.
    """

    prompts: List[str]
    max_tokens: int = 32  # the opening of the answer: divergence compounds over a long greedy completion
    tolerance: float = 0.02  # --max-quality-loss: drift ABOVE the reference's own noise before a precision goes
    timeout: float = 120.0
    answers: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    repeats: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)  # a second pass, for the noise floor
    notes: List[str] = field(default_factory=list)

    def wanted(self, backend: str, quant: str) -> bool:
        return bool(self.prompts) and (backend, quant) not in self.answers

    def record(self, base_url: str, backend: str, quant: str) -> None:
        """Answer the prompts twice on this running engine, unless this precision has already been measured.

        The second pass is what makes the first interpretable: identical settings, same prompts, so anything that
        differs between them is the engine's own noise rather than the weights.
        """
        if not self.wanted(backend, quant):
            return
        first = greedy_answers(base_url, self.prompts, self.max_tokens, self.timeout)
        again = greedy_answers(base_url, self.prompts, self.max_tokens, self.timeout)
        if not _answered(first) or not _answered(again):
            # Every request failed. Empty answers compare equal to any other empty set, so recording them would
            # report perfect agreement and quietly wave the precision through; record nothing and let a later
            # trial of this precision try again.
            logger.warning("quality probe got no answers from %s/%s: not recording, so it gates nothing",
                           backend, quant)
            return
        self.answers[(backend, quant)] = first
        self.repeats[(backend, quant)] = again

    def noise(self, backend: str) -> Optional[float]:
        """How much the reference precision disagrees with itself: the floor any real drift must clear."""
        ref = self.reference_for(backend)
        if ref is None:
            return None
        first, again = self.answers.get((backend, ref)), self.repeats.get((backend, ref))
        if not first or not again:
            return None
        return 1.0 - agreement(first, again)

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
        """Why this precision should be dropped, or None to keep it.

        The bar is the reference's own noise plus the tolerance. Identical weights answering the same prompts
        twice already disagree sometimes — batched decoding is not bit-reproducible — and a gate that ignored
        that would refuse every precision, including the reference.
        """
        d = self.drift(backend, quant)
        if d is None:
            return None
        floor = self.noise(backend) or 0.0
        if d <= floor + self.tolerance:
            return None
        ref = self.reference_for(backend)
        return (f"{backend}/{quant} answered {d:.0%} of {len(self.prompts)} prompts differently from "
                f"{backend}/{ref}, over the {floor + self.tolerance:.0%} allowed "
                f"({self.tolerance:.0%} on top of the {floor:.0%} that {backend}/{ref} differs from itself)")

    def summary(self) -> List[str]:
        """One note per measured precision, for the profile: what drifted, against what floor."""
        lines: List[str] = []
        for (backend, quant) in sorted(self.answers):
            d = self.drift(backend, quant)
            if d is None:
                floor = self.noise(backend)
                floor_text = f", which differs from itself on {floor:.0%}" if floor is not None else ""
                lines.append(f"quality reference {backend}/{quant}: the first {self.max_tokens} tokens of "
                             f"{len(self.prompts)} answers, greedy{floor_text}")
            else:
                lines.append(f"quality {backend}/{quant}: {d:.0%} of answers differ from "
                             f"{backend}/{self.reference_for(backend)} (allowed {(self.noise(backend) or 0.0) + self.tolerance:.0%})")
        return lines
