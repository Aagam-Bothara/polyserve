from __future__ import annotations

from polyserve.backends.llamacpp import LlamaCppCudaBackend
from polyserve.backends.vllm import PAGED_KV_FRACTION, VllmBackend
from polyserve.memory import MemoryModel, estimate, kv_cache_bytes, plan, safety_margin
from polyserve.models import Config, GiB, MiB


def test_safety_margin_rule():
    assert safety_margin(4 * GiB) == 512 * MiB  # 5% of 4 GiB = 204 MiB < 512 MiB
    assert safety_margin(80 * GiB) == int(0.05 * 80 * GiB)


def test_estimate_components_match_formula(hw_a100, prepared_vllm):
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    mm = MemoryModel(runtime_workspace=1 * GiB, kv_tokens_fn=lambda c: c.ctx * c.batch, device="gpu")
    est = estimate(hw_a100, prepared_vllm, cfg, mm)
    assert est.weights == prepared_vllm.weights_bytes["bf16"]
    assert est.kv_cache == 114_688 * 4096 * 64
    assert est.runtime_workspace == 1 * GiB
    assert est.safety_margin == safety_margin(hw_a100.gpu.vram_free_bytes)
    assert est.total == est.weights + est.kv_cache + est.runtime_workspace + est.safety_margin
    assert est.feasible


def test_budget_is_capped_by_gpu_memory_utilization(hw_rtx4090, prepared_vllm):
    cfg = Config(backend="vllm", quant="bf16", ctx=2048, batch=16, gpu_memory_utilization=0.5)
    mm = VllmBackend().memory_model(hw_rtx4090)
    est = estimate(hw_rtx4090, prepared_vllm, cfg, mm)
    assert est.budget == int(0.5 * hw_rtx4090.gpu.vram_total_bytes)


def test_plan_prunes_the_grid(hw_rtx4090, prepared_vllm):
    be = VllmBackend()
    grid = be.candidate_configs(hw_rtx4090, prepared_vllm)
    kept = plan(hw_rtx4090, prepared_vllm, grid, be.memory_model(hw_rtx4090))
    assert 0 < len(kept) < len(grid)
    # The biggest configs (ctx 8192 x batch 256) cannot fit 3B bf16 + KV on 24 GB.
    assert not any(c.ctx == 8192 and c.batch == 256 and c.quant == "bf16" for c, _ in kept)
    # Small ones do.
    assert any(c.ctx == 2048 and c.batch == 16 for c, _ in kept)


def test_almost_everything_fits_on_a100(hw_a100, prepared_vllm):
    be = VllmBackend()
    grid = be.candidate_configs(hw_a100, prepared_vllm)
    kept = plan(hw_a100, prepared_vllm, grid, be.memory_model(hw_a100))
    # Only ctx 8192 x batch 256 at the lowest gpu_memory_utilization (0.80 x 80 GiB) exceeds the budget.
    dropped = [c for c in grid if c.key() not in {k.key() for k, _ in kept}]
    assert len(dropped) == 2
    assert all(c.ctx == 8192 and c.batch == 256 and c.gpu_memory_utilization == 0.80 for c in dropped)
    assert any(c.ctx == 4096 and c.batch == 256 and c.quant == "bf16" for c, _ in kept)


def test_paged_kv_fraction_applied(hw_a100, prepared_vllm):
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    mm = VllmBackend().memory_model(hw_a100)
    assert mm.kv_tokens_fn(cfg) == int(4096 * 64 * PAGED_KV_FRACTION)


def test_partial_offload_scales_device_memory_and_checks_host(hw_gtx1080, prepared_llamacpp):
    be = LlamaCppCudaBackend()
    mm = be.memory_model(hw_gtx1080)
    full = Config(backend="llamacpp-cuda", quant="Q8_0", ctx=2048, batch=1, n_gpu_layers=29)
    half = Config(backend="llamacpp-cuda", quant="Q8_0", ctx=2048, batch=1, n_gpu_layers=14)
    e_full = estimate(hw_gtx1080, prepared_llamacpp, full, mm)
    e_half = estimate(hw_gtx1080, prepared_llamacpp, half, mm)
    assert e_half.weights == int(prepared_llamacpp.weights_bytes["Q8_0"] * 14 / 28)
    assert e_half.total < e_full.total
    # Host RAM too small for the remainder -> dropped by plan().
    hw_gtx1080.cpu.ram_free_bytes = 512 * MiB
    assert plan(hw_gtx1080, prepared_llamacpp, [half], mm) == []


def test_llamacpp_kv_is_full_context_times_slots(hw_gtx1080, prepared_llamacpp):
    cfg = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=29)
    mm = LlamaCppCudaBackend().memory_model(hw_gtx1080)
    assert kv_cache_bytes(prepared_llamacpp, cfg, mm.kv_tokens_fn(cfg)) == 114_688 * 4096 * 4


def test_cpu_planner_uses_ram(hw_cpu, prepared_llamacpp):
    from polyserve.backends.llamacpp import LlamaCppCpuBackend

    be = LlamaCppCpuBackend()
    prepared_llamacpp.backend = "llamacpp-cpu"
    grid = be.candidate_configs(hw_cpu, prepared_llamacpp)
    kept = plan(hw_cpu, prepared_llamacpp, grid, be.memory_model(hw_cpu))
    assert kept and all(e.budget == int(0.95 * hw_cpu.cpu.ram_free_bytes) for _, e in kept)
