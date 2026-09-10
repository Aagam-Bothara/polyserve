"""Memory planner (runs before any benchmark).

    estimated = weights + kv_cache(ctx, batch, dtype) + runtime_workspace + safety_margin
    keep config only if estimated <= 0.95 * available_memory
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from polyserve.hfconfig import dtype_bytes
from polyserve.models import Config, HardwareDescriptor, MemoryEstimate, MiB, PreparedModel

logger = logging.getLogger(__name__)

BUDGET_FRACTION = 0.95
MIN_SAFETY_MARGIN = 512 * MiB


def safety_margin(available: int) -> int:
    """5% or 512 MB, whichever is larger."""
    return max(int(0.05 * available), MIN_SAFETY_MARGIN)


def kv_cache_bytes(model: PreparedModel, cfg: Config, kv_tokens: int) -> int:
    kv_dtype = cfg.kv_dtype if cfg.kv_dtype != "auto" else model.arch.torch_dtype
    return model.arch.kv_bytes_per_token(dtype_bytes(kv_dtype)) * kv_tokens


class MemoryModel:
    """Backend-provided knobs for the generic estimator."""

    def __init__(self, runtime_workspace: int, kv_tokens_fn, device: str = "gpu"):
        self.runtime_workspace = runtime_workspace
        self.kv_tokens_fn = kv_tokens_fn  # Config -> int tokens resident in KV cache
        self.device = device  # "gpu" | "cpu"


def estimate(
    hw: HardwareDescriptor,
    model: PreparedModel,
    cfg: Config,
    mm: MemoryModel,
    available: Optional[int] = None,
) -> MemoryEstimate:
    weights = model.weights_for(cfg.quant)
    kv = kv_cache_bytes(model, cfg, mm.kv_tokens_fn(cfg))

    # Partial GPU offload (llama.cpp): only the offloaded fraction lands on the device.
    if mm.device == "gpu" and cfg.n_gpu_layers is not None:
        frac = min(1.0, max(0.0, cfg.n_gpu_layers / max(model.arch.num_layers, 1)))
        weights = int(weights * frac)
        kv = int(kv * frac)

    if available is None:
        available = hw.gpu.vram_free_bytes if (mm.device == "gpu" and hw.gpu) else hw.cpu.ram_free_bytes
    margin = safety_margin(available)
    total = weights + kv + mm.runtime_workspace + margin

    # vLLM/SGLang cap their own allocation at gpu_memory_utilization x total VRAM.
    budget = int(BUDGET_FRACTION * available)
    if cfg.gpu_memory_utilization is not None and hw.gpu is not None and mm.device == "gpu":
        budget = min(budget, int(cfg.gpu_memory_utilization * hw.gpu.vram_total_bytes))
        # ... and cannot use more than what is actually free right now.
        budget = min(budget, int(BUDGET_FRACTION * available))

    return MemoryEstimate(
        config_key=cfg.key(),
        weights=weights,
        kv_cache=kv,
        runtime_workspace=mm.runtime_workspace,
        safety_margin=margin,
        total=total,
        budget=budget,
        feasible=total <= budget,
    )


def plan(
    hw: HardwareDescriptor,
    model: PreparedModel,
    configs: Sequence[Config],
    mm: MemoryModel,
) -> List[Tuple[Config, MemoryEstimate]]:
    """Return (config, estimate) pairs that fit. Also checks host RAM for CPU-offloaded layers."""
    kept: List[Tuple[Config, MemoryEstimate]] = []
    for cfg in configs:
        est = estimate(hw, model, cfg, mm)
        if not est.feasible:
            logger.debug("drop %s: %.2f GiB > budget %.2f GiB", cfg.key(), est.total / 2**30, est.budget / 2**30)
            continue
        if mm.device == "gpu" and cfg.n_gpu_layers is not None and cfg.n_gpu_layers < model.arch.num_layers:
            # Remainder lives in host RAM; make sure that fits too.
            frac_cpu = 1.0 - cfg.n_gpu_layers / max(model.arch.num_layers, 1)
            host_need = int(model.weights_for(cfg.quant) * frac_cpu) + safety_margin(hw.cpu.ram_free_bytes)
            if host_need > BUDGET_FRACTION * hw.cpu.ram_free_bytes:
                logger.debug("drop %s: host RAM insufficient for offload remainder", cfg.key())
                continue
        kept.append((cfg, est))
    return kept
