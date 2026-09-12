"""Removing a memory-saving strategy steps the batch down to what would have fitted without it."""

from __future__ import annotations

from types import SimpleNamespace

import polyserve.bench.ablation as AB
from polyserve.backends import get_backend
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config


def test_int4_removal_steps_the_batch_down_to_fit(monkeypatch, hw_a100, prepared_vllm):
    # Measured on an A40 (chat-system): fp8 weights at the 4-bit pick's batch 256 did not fit.
    monkeypatch.setattr(AB, "estimate",
                        lambda hw, model, c, mm: SimpleNamespace(feasible=c.quant == "awq" or c.batch <= 64))
    model = prepared_vllm.model_copy(update={"weights_bytes": {**prepared_vllm.weights_bytes, "awq": 2_000_000_000}})
    pick = Config(backend="vllm", quant="awq", ctx=16384, batch=256, gpu_memory_utilization=0.95)
    (w,) = [v for v in AB.strategy_variants(pick, get_backend("vllm"), hw_a100, model, get_workload("chat"))
            if v.strategy == "weights"]
    assert w.label == "-int4 (fp8)" and w.config.quant == "fp8" and w.config.batch == 64
    assert w.note == "batch 64: fp8 weights does not fit at batch 256"
