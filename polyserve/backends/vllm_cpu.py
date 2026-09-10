"""vLLM-CPU backend (x86 with AVX512). Same server module as vLLM, CPU build of the package."""

from __future__ import annotations

import math
import sys
from typing import List

from polyserve.backends.base import LaunchSpec, LlmtraceHooks
from polyserve.backends.vllm import PAGED_KV_FRACTION, VllmBackend
from polyserve.hfconfig import load_arch
from polyserve.memory import MemoryModel, kv_cache_bytes
from polyserve.models import Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel


class VllmCpuBackend(VllmBackend):
    name = "vllm-cpu"
    runtime_workspace_bytes = int(1.0 * GiB)

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if not hw.cpu.avx512:
            return False
        try:
            load_arch(model)
        except Exception:
            return False
        return True

    def precisions(self, hw: HardwareDescriptor) -> List[str]:
        return ["bf16"]

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return MemoryModel(
            runtime_workspace=self.runtime_workspace_bytes,
            kv_tokens_fn=lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION),
            device="cpu",
        )

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel) -> List[Config]:
        max_pos = model.arch.max_position_embeddings
        ctxs = [c for c in (2048, 4096) if c <= max_pos] or [max_pos]
        return [
            Config(backend=self.name, quant="bf16", ctx=ctx, batch=batch)
            for ctx in ctxs
            for batch in (4, 16, 64)
        ]

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel) -> Config:
        return Config(backend=self.name, quant="bf16", ctx=min(4096, model.arch.max_position_embeddings), batch=16)

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        args = [
            sys.executable, "-m", self.server_module,
            "--model", model.hf_path or model.spec.hf_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-model-len", str(cfg.ctx),
            "--max-num-seqs", str(cfg.batch),
            "--dtype", "bfloat16",
            "--disable-log-requests",
        ]
        if model.spec.revision:
            args += ["--revision", model.spec.revision]
        for k, v in cfg.extra.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        kv_gib = max(4, math.ceil(kv_cache_bytes(model, cfg, int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION)) / GiB))
        env = {
            "VLLM_CPU_KVCACHE_SPACE": str(kv_gib),
            "VLLM_CPU_OMP_THREADS_BIND": "auto",
            "VLLM_TARGET_DEVICE": "cpu",
        }
        return LaunchSpec(args=args, env=env)

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(health_path="/health", model_name=model.spec.hf_id, gpu_ids=[], process_memory=True)
