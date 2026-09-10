"""llama.cpp backend: `llama-server`, CUDA (any NVIDIA GPU, incl. Pascal/Volta) and CPU."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from polyserve.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks
from polyserve.gguf import (
    GGUF_QUANTS,
    GGUFCandidate,
    convert_and_quantize,
    download_gguf,
    estimate_gguf_bytes,
    search_hub_gguf,
)
from polyserve.hardware import llama_server_binary
from polyserve.hfconfig import load_arch
from polyserve.memory import MemoryModel
from polyserve.models import Config, HardwareDescriptor, MiB, ModelSpec, PreparedModel

logger = logging.getLogger(__name__)


class LlamaCppBackend(BaseBackend):
    """Shared implementation; the CUDA and CPU variants differ in offload and workspace."""

    name = "llamacpp"
    cuda = False
    runtime_workspace_bytes = 768 * MiB  # compute buffers (scale with n_batch) + CUDA context

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if self.cuda and (hw.gpu is None or hw.gpu.vendor != "nvidia"):
            return False
        return llama_server_binary() is not None

    # ---- prepare: resolve (no download); materialize: download/convert only what the planner kept

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel:
        arch = load_arch(model)
        quants = list(quants or GGUF_QUANTS)
        prepared = PreparedModel(spec=model, backend=self.name, arch=arch)
        if os.path.isfile(model.hf_id) and model.hf_id.lower().endswith(".gguf"):
            # Local GGUF file given directly.
            q = _quant_from_filename(model.hf_id) or "Q4_K_M"
            prepared.gguf_paths[q] = model.hf_id
            prepared.weights_bytes[q] = os.path.getsize(model.hf_id)
            return prepared
        try:
            found: Dict[str, GGUFCandidate] = search_hub_gguf(model, quants)
        except Exception as exc:
            logger.warning("GGUF hub search failed (%s); will fall back to conversion", exc)
            found = {}
        params = arch.num_params or 0
        for q in quants:
            cand = found.get(q)
            if cand is not None:
                prepared.weights_bytes[q] = cand.size_bytes or estimate_gguf_bytes(params, q)
                prepared.gguf_paths[q] = f"hf://{cand.repo_id}/{cand.filename}"
            else:
                prepared.weights_bytes[q] = estimate_gguf_bytes(params, q)
                prepared.gguf_paths[q] = f"convert://{q}"
        return prepared

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        need_convert: List[str] = []
        for q in quants:
            ref = model.gguf_paths.get(q)
            if ref is None:
                continue
            if ref.startswith("hf://"):
                repo, _, fname = ref[len("hf://"):].rpartition("/")
                path = download_gguf(GGUFCandidate(repo_id=repo, filename=fname, quant=q))
                model.gguf_paths[q] = str(path)
                model.weights_bytes[q] = path.stat().st_size
            elif ref.startswith("convert://"):
                need_convert.append(q)
        if need_convert:
            produced = convert_and_quantize(model.spec, need_convert)
            for q, path in produced.items():
                model.gguf_paths[q] = str(path)
                model.weights_bytes[q] = Path(path).stat().st_size
            for q in need_convert:
                if q not in produced:
                    model.gguf_paths.pop(q, None)
                    model.weights_bytes.pop(q, None)
        return model

    # ---- memory: llama-server allocates the full KV for -c up front (per-slot ctx x n_parallel)

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return MemoryModel(
            runtime_workspace=self.runtime_workspace_bytes,
            kv_tokens_fn=lambda cfg: cfg.ctx * cfg.batch,
            device="gpu" if self.cuda else "cpu",
        )

    # ---- configs

    def _quants(self, model: PreparedModel) -> List[str]:
        return [q for q in GGUF_QUANTS if q in model.weights_bytes] or list(model.weights_bytes)

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel) -> List[Config]:
        layers = model.arch.num_layers
        max_pos = model.arch.max_position_embeddings
        ctxs = [c for c in (2048, 4096, 8192) if c <= max_pos] or [max_pos]
        if self.cuda:
            ngls = sorted({layers + 1, (3 * layers) // 4, layers // 2}, reverse=True)  # +1 = output layer too
            n_batches = [512]
        else:
            ngls = [0]
            n_batches = [512, 2048]
        out: List[Config] = []
        for quant in self._quants(model):
            for ngl in ngls:
                for ctx in ctxs:
                    for np_ in (1, 4, 8):
                        for nb in n_batches:
                            out.append(
                                Config(
                                    backend=self.name, quant=quant, ctx=ctx, batch=np_, n_gpu_layers=ngl, n_batch=nb
                                )
                            )
        return out

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel) -> Config:
        quants = self._quants(model)
        return Config(
            backend=self.name,
            quant="Q4_K_M" if "Q4_K_M" in quants else quants[0],
            ctx=min(4096, model.arch.max_position_embeddings),
            batch=4,
            n_gpu_layers=(model.arch.num_layers + 1) if self.cuda else 0,
            n_batch=512,
        )

    # ---- launch

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        binary = llama_server_binary()
        if binary is None:
            raise RuntimeError("llama-server not found on PATH (set $LLAMA_SERVER)")
        path = model.gguf_paths.get(cfg.quant)
        if not path or path.startswith(("hf://", "convert://")):
            raise RuntimeError(f"GGUF for {cfg.quant} not materialized: {path}")
        n_parallel = max(1, cfg.batch)
        args = [
            binary,
            "-m", path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(cfg.ctx * n_parallel),  # -c is total context, split across slots
            "-np", str(n_parallel),
            "-b", str(cfg.n_batch or 512),
            "-ngl", str(cfg.n_gpu_layers if cfg.n_gpu_layers is not None else (999 if self.cuda else 0)),
            "--alias", model.spec.hf_id,
        ]
        threads = cfg.extra.get("threads")
        if threads:
            args += ["-t", str(threads)]
        if self.cuda:
            args += ["-fa", "on"]
        if cfg.kv_dtype != "auto":
            args += ["-ctk", cfg.kv_dtype, "-ctv", cfg.kv_dtype]
        for k, v in cfg.extra.items():
            if k == "threads":
                continue
            args += [f"--{k.replace('_', '-')}", str(v)]
        env = {}
        if not self.cuda:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return LaunchSpec(args=args, env=env)

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/health",
            model_name=model.spec.hf_id,
            tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if (self.cuda and hw.gpu) else [],
            process_memory=not self.cuda,
        )


class LlamaCppCudaBackend(LlamaCppBackend):
    name = "llamacpp-cuda"
    cuda = True


class LlamaCppCpuBackend(LlamaCppBackend):
    name = "llamacpp-cpu"
    cuda = False
    runtime_workspace_bytes = 512 * MiB

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel) -> List[Config]:
        cfgs = super().candidate_configs(hw, model)
        for c in cfgs:
            c.extra["threads"] = hw.cpu.physical_cores
        return cfgs

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel) -> Config:
        c = super().default_config(hw, model)
        c.extra["threads"] = hw.cpu.physical_cores
        return c


def _quant_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path).upper()
    for q in GGUF_QUANTS + ("F16", "BF16", "F32"):
        if q in base:
            return q
    return None
