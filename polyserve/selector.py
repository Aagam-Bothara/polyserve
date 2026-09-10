"""Step 2: select candidate backends from the hardware descriptor.

Mirrors the rules in the spec exactly; calibration picks the winner among them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from polyserve.models import HardwareDescriptor, ModelSpec

logger = logging.getLogger(__name__)

MIN_CC_MODERN = (7, 5)


def select_backends(
    hw: HardwareDescriptor,
    model: ModelSpec,
    registry: Dict[str, "object"],
    force: Optional[str] = None,
) -> List[str]:
    """Return backend names in preference order. `registry` maps name -> Backend instance.

    The registry entry's `supports(hw, model)` is consulted for the
    "model_in_<backend>_registry" checks from the spec.
    """
    if force:
        if force not in registry:
            raise ValueError(f"unknown backend {force!r}; known: {sorted(registry)}")
        return [force]

    candidates: List[str] = []
    gpu = hw.gpu

    def ok(name: str) -> bool:
        b = registry.get(name)
        if b is None or not hw.backend_available(name):
            return False
        try:
            return bool(b.supports(hw, model))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("%s.supports() raised %s; excluding", name, exc)
            return False

    if gpu is not None and gpu.vendor == "nvidia":
        if gpu.cc >= MIN_CC_MODERN:
            if ok("vllm"):
                candidates.append("vllm")
            if ok("sglang"):
                candidates.append("sglang")
        if ok("llamacpp-cuda"):
            candidates.append("llamacpp-cuda")
    if gpu is None:
        if ok("llamacpp-cpu"):
            candidates.append("llamacpp-cpu")
        if hw.cpu.avx512 and ok("vllm-cpu"):
            candidates.append("vllm-cpu")
    elif not candidates:
        # GPU present but no GPU backend usable (e.g. non-NVIDIA, or no CUDA build installed):
        # fall back to CPU so `polyserve serve` still works.
        if ok("llamacpp-cpu"):
            candidates.append("llamacpp-cpu")
    return candidates


def explain(hw: HardwareDescriptor) -> List[str]:
    """Human-readable reasons for each backend's availability."""
    lines = []
    for name, b in sorted(hw.backends.items()):
        state = "available" if b.available else f"unavailable ({b.reason or 'unknown'})"
        ver = f" v{b.version}" if b.version else ""
        lines.append(f"{name}{ver}: {state}")
    return lines
