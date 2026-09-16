"""Does live traffic still look like the traffic this profile was tuned on?

A profile is calibrated for one shape of work — prompt length, answer length, concurrency — and nothing re-tunes
itself when real traffic drifts away from that shape. The proxy feeds every request through here, and the watch
says in one report whether what is being served still resembles what was measured, so `polyserve recalibrate`
is a decision rather than a guess.

Only what the backend already reports is counted: the `usage` block of non-streaming responses. Streamed replies
are passed through byte-for-byte, so they count as requests and toward concurrency but carry no token counts; a
purely streaming deployment therefore sees concurrency drift but not length drift. Nothing here reads prompt text.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence

WINDOW = 512  # requests kept for the rolling picture
MIN_REQUESTS = 50  # below this, say "not enough traffic yet" rather than cry drift
WIDER_THAN = 1.5  # a median this many times the calibrated value counts as drift
NARROWER_THAN = 1 / WIDER_THAN


def _median(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _percentile(values: Sequence[int], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return float(ordered[idx])


class TrafficWatch:
    """Rolling picture of served traffic, against the workload the profile was calibrated on."""

    def __init__(self, workload_spec: Optional[Dict[str, Any]] = None, window: int = WINDOW,
                 min_requests: int = MIN_REQUESTS) -> None:
        spec = workload_spec or {}
        self.workload_name = spec.get("name")
        self.calibrated_prompt = spec.get("prefill_tokens")
        self.calibrated_completion = spec.get("decode_tokens")
        self.calibrated_concurrency: List[int] = [int(c) for c in (spec.get("concurrencies") or [])]
        self.min_requests = min_requests
        self.requests = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.prompt_tokens: Deque[int] = deque(maxlen=window)
        self.completion_tokens: Deque[int] = deque(maxlen=window)
        self.concurrency: Deque[int] = deque(maxlen=window)
        self._warned = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ recording

    def began(self) -> None:
        with self._lock:
            self.requests += 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.concurrency.append(self.in_flight)

    def ended(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    def record_response(self, content: bytes, media_type: str = "") -> None:
        """Take the token counts out of a non-streaming response body, if it carries a usage block."""
        if "json" not in media_type.lower() or not content:
            return
        try:
            body = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            return
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return
        with self._lock:
            for key, into in (("prompt_tokens", self.prompt_tokens), ("completion_tokens", self.completion_tokens)):
                value = usage.get(key)
                if isinstance(value, int) and value > 0:
                    into.append(value)

    # ------------------------------------------------------------------ reading

    @property
    def enough_data(self) -> bool:
        return self.requests >= self.min_requests

    def findings(self) -> List[str]:
        """Plain sentences about how live traffic differs from the calibrated workload; empty when it matches."""
        out: List[str] = []
        if not self.enough_data:
            return out
        for label, seen, calibrated in (("prompts", _median(self.prompt_tokens), self.calibrated_prompt),
                                        ("answers", _median(self.completion_tokens), self.calibrated_completion)):
            if seen is None or not calibrated:
                continue
            ratio = seen / calibrated
            if ratio >= WIDER_THAN or ratio <= NARROWER_THAN:
                longer = "longer" if ratio > 1 else "shorter"
                out.append(f"{label} are {max(ratio, 1 / ratio):.1f}x {longer} than the profile was tuned for "
                           f"(median {seen:.0f} tokens against {calibrated})")
        typical = _median(self.concurrency)
        if typical is not None and self.calibrated_concurrency:
            top = max(self.calibrated_concurrency)
            if typical > top:
                levels = "/".join(str(c) for c in self.calibrated_concurrency)
                out.append(f"traffic runs at {typical:.0f} concurrent requests; the profile was measured at {levels}")
        return out

    def should_warn(self) -> bool:
        """True once, the first time drift is worth telling the operator about."""
        with self._lock:
            if self._warned or not self.findings():
                return False
            self._warned = True
            return True

    def report(self) -> Dict[str, Any]:
        findings = self.findings()
        return {
            "calibrated_workload": self.workload_name,
            "requests_seen": self.requests,
            "requests_with_token_counts": len(self.completion_tokens),
            "enough_data": self.enough_data,
            "prompt_tokens": {"median": _median(self.prompt_tokens), "p90": _percentile(self.prompt_tokens, 90),
                              "calibrated": self.calibrated_prompt},
            "completion_tokens": {"median": _median(self.completion_tokens),
                                  "p90": _percentile(self.completion_tokens, 90),
                                  "calibrated": self.calibrated_completion},
            "concurrency": {"median": _median(self.concurrency), "peak": self.peak_in_flight,
                            "calibrated": self.calibrated_concurrency},
            "drift": findings,
            "verdict": ("not enough traffic yet" if not self.enough_data else
                        "drifted from the calibrated workload" if findings else
                        "matches the calibrated workload"),
        }
