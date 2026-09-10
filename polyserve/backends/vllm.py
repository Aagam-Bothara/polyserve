"""vLLM backend (NVIDIA, compute capability >= 7.5)."""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import List, Optional

from polyserve.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid
from polyserve.hfconfig import dtype_bytes, load_arch
from polyserve.memory import MemoryModel
from polyserve.models import Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel

logger = logging.getLogger(__name__)

# Architectures known to load in vLLM; consulted only when the live registry is unavailable.
KNOWN_ARCHS = {
    "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM", "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration", "Phi3ForCausalLM", "PhiForCausalLM", "GPT2LMHeadModel", "GPTNeoXForCausalLM",
    "OPTForCausalLM", "FalconForCausalLM", "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "OlmoForCausalLM",
    "Olmo2ForCausalLM", "InternLM2ForCausalLM", "StableLmForCausalLM", "CohereForCausalLM", "GraniteForCausalLM",
    "Starcoder2ForCausalLM", "BloomForCausalLM", "MPTForCausalLM", "ExaoneForCausalLM", "SmolLM3ForCausalLM",
}

# Paged-KV runtimes do not need KV for every (ctx x batch) token up front; they preempt when
# occupancy is exceeded. Budget a fraction of the worst case so realistic batches survive planning.
PAGED_KV_FRACTION = 0.25


def vllm_registry_archs() -> Optional[set]:
    if importlib.util.find_spec("vllm") is None:
        return None
    try:
        from vllm.model_executor.models import ModelRegistry  # type: ignore

        return set(ModelRegistry.get_supported_archs())
    except Exception as exc:
        logger.debug("vllm registry unavailable: %s", exc)
        return None


class VllmBackend(BaseBackend):
    name = "vllm"
    runtime_workspace_bytes = int(1.5 * GiB)  # CUDA context + CUDA graphs + activation workspace
    server_module = "vllm.entrypoints.openai.api_server"

    # ---- capability

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if hw.gpu is None or hw.gpu.vendor != "nvidia" or hw.gpu.cc < (7, 5):
            return False
        try:
            arch = load_arch(model).architecture
        except Exception as exc:
            logger.warning("could not read config for %s: %s", model.hf_id, exc)
            return True
        live = vllm_registry_archs()
        if live is not None:
            return arch in live
        return arch in KNOWN_ARCHS or arch == "unknown"

    # ---- prepare

    def precisions(self, hw: HardwareDescriptor) -> List[str]:
        cc = hw.gpu.cc if hw.gpu else (0, 0)
        out = ["bf16" if cc >= (8, 0) else "fp16"]
        if cc >= (8, 0):  # fp8 online weight quantisation (Marlin on Ampere, native on Ada/Hopper)
            out.append("fp8")
        return out

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel:
        arch = load_arch(model)
        params = arch.num_params or 0
        weights = {}
        for p in quants or self.precisions(hw):
            weights[p] = params * (1 if p == "fp8" else dtype_bytes(p))
        return PreparedModel(spec=model, backend=self.name, arch=arch, hf_path=model.hf_id, weights_bytes=weights)

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        return model  # vLLM pulls HF weights itself at launch

    # ---- memory

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return MemoryModel(
            runtime_workspace=self.runtime_workspace_bytes,
            kv_tokens_fn=lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION),
            device="gpu",
        )

    # ---- configs

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]:
        ctxs = ctx_grid(model.arch.max_position_embeddings, min_ctx)
        out: List[Config] = []
        for quant in [q for q in self.precisions(hw) if q in model.weights_bytes]:
            for gmu in (0.80, 0.90, 0.95):
                for ctx in ctxs:
                    for batch in (16, 64, 256):
                        out.append(
                            Config(backend=self.name, quant=quant, ctx=ctx, batch=batch, gpu_memory_utilization=gmu)
                        )
        return out

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config:
        return Config(
            backend=self.name,
            quant=self.precisions(hw)[0],
            ctx=min(max(4096, min_ctx), model.arch.max_position_embeddings),
            batch=256,
            gpu_memory_utilization=0.90,
        )

    # ---- launch

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        args = [
            sys.executable, "-m", self.server_module,
            "--model", model.hf_path or model.spec.hf_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-model-len", str(cfg.ctx),
            "--max-num-seqs", str(cfg.batch),
        ]
        if cfg.gpu_memory_utilization is not None:
            args += ["--gpu-memory-utilization", f"{cfg.gpu_memory_utilization:.2f}"]
        if cfg.quant == "fp8":
            args += ["--quantization", "fp8"]
        elif cfg.quant in ("bf16", "fp16"):
            args += ["--dtype", "bfloat16" if cfg.quant == "bf16" else "float16"]
        if cfg.kv_dtype != "auto":
            args += ["--kv-cache-dtype", cfg.kv_dtype]
        if model.spec.revision:
            args += ["--revision", model.spec.revision]
        for k, v in cfg.extra.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        return LaunchSpec(args=args)

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/health",
            model_name=model.hf_path or model.spec.hf_id,
            tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if hw.gpu else [],
        )
