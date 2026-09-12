"""vLLM backend (NVIDIA, compute capability >= 7.5)."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from typing import List, Optional, Tuple

from polyserve import speculative
from polyserve.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from polyserve.hfconfig import dtype_bytes, load_arch
from polyserve.memory import MemoryModel
from polyserve.models import Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel
from polyserve.hardware import nvlink_between
from polyserve.quantized import INT4_METHODS, hf_weight_options

# The fp8 KV-cache type offered on Ampere. It runs through FlashInfer: vLLM 0.11's default Triton
# attention builds its fp8 kernels with e4m3, which Ampere cannot compile, whatever the cache type.
AMPERE_FP8_KV = "fp8_e5m2"

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
        if cc >= (7, 5):  # pre-quantized 4-bit checkpoints: Marlin kernels on Ampere+, plain AWQ/GPTQ on Turing
            out += list(INT4_METHODS)
        return out

    def supported_quants(self, hw: HardwareDescriptor) -> List[str]:
        return self.precisions(hw)

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel:
        arch = load_arch(model)
        weights, paths = hf_weight_options(model, arch.num_params or 0, list(quants or self.precisions(hw)),
                                           lambda p: 1 if p == "fp8" else dtype_bytes(p))
        return PreparedModel(spec=model, backend=self.name, arch=arch, hf_path=model.hf_id, hf_paths=paths,
                             weights_bytes=weights)

    # ---- optional search dimensions

    supports_tp = True

    def kv_dtypes(self, hw: HardwareDescriptor) -> List[str]:
        cc = hw.gpu.cc if hw.gpu else (0, 0)
        if cc >= (8, 9):  # Ada, Hopper: e4m3
            return ["fp8"]
        if cc >= (8, 0) and importlib.util.find_spec("flashinfer") is not None:  # Ampere, via FlashInfer
            return [AMPERE_FP8_KV]
        return []

    def batch_ladder(self) -> Tuple[int, ...]:
        return (16, 64, 256, 512)

    def spec_variants(self, cfg: Config, model: PreparedModel) -> List[Config]:
        specs = [speculative.ngram()]
        draft = speculative.draft_for(model.spec.hf_id)
        if draft and speculative.vllm_supports_draft():
            specs.append(speculative.draft(draft, speculative.DRAFT_TOKENS_VLLM))
        return [cfg.model_copy(update={"spec_decode": s}) for s in specs if s != cfg.spec_decode]

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        return model  # vLLM pulls HF weights itself at launch

    # ---- memory

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return self.calibrated_memory(hw, lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION), "gpu")

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
            "--model", model.hf_paths.get(cfg.quant) or model.hf_path or model.spec.hf_id,
            # A stable name even when a 4-bit checkpoint repository is what gets loaded.
            "--served-model-name", model.spec.hf_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-model-len", str(cfg.ctx),
            "--max-num-seqs", str(cfg.batch),
        ]
        if cfg.gpu_memory_utilization is not None:
            args += ["--gpu-memory-utilization", f"{cfg.gpu_memory_utilization:.2f}"]
        if cfg.prefill_budget is not None:
            args += ["--max-num-batched-tokens", str(cfg.prefill_budget)]
        if cfg.quant == "fp8":
            args += ["--quantization", "fp8"]
        elif cfg.quant in ("bf16", "fp16"):
            args += ["--dtype", "bfloat16" if cfg.quant == "bf16" else "float16"]
        # awq / gptq: the checkpoint's quantization_config selects the kernel; no flag needed.
        if cfg.kv_dtype != "auto":
            args += ["--kv-cache-dtype", cfg.kv_dtype]
        if cfg.prefix_cache is not None:
            args += ["--enable-prefix-caching" if cfg.prefix_cache else "--no-enable-prefix-caching"]
        if cfg.spec_decode:
            args += ["--speculative-config", json.dumps(speculative.vllm_config(cfg.spec_decode))]
        if cfg.tp > 1:
            args += ["--tensor-parallel-size", str(cfg.tp)]
        if model.spec.revision and cfg.quant not in INT4_METHODS:
            args += ["--revision", model.spec.revision]
        args += render_extra(cfg.extra)
        env = {"VLLM_ATTENTION_BACKEND": "FLASHINFER"} if cfg.kv_dtype == AMPERE_FP8_KV else {}
        # PCIe-only GPUs: peer-to-peer can hang at start-up inside containers. Measured on a pair of
        # A40s: NCCL's P2P path hangs at init, and with only that disabled the engine's custom
        # all-reduce (CUDA IPC, also P2P) hangs next. With both off it starts in under a minute.
        if cfg.tp > 1 and nvlink_between(list(range(cfg.tp))) is False:
            env["NCCL_P2P_DISABLE"] = "1"
            args += ["--disable-custom-all-reduce"]
        return LaunchSpec(args=args, env=env)

    # Chunked-prefill token budgets to try. Small budgets interleave prefill with decode and protect
    # per-token latency; large ones finish long prompts in fewer steps and cut time to first token.
    PREFILL_BUDGETS = (2048, 8192, 16384)

    def prefill_variants(self, cfg: Config) -> List[Config]:
        # vLLM requires max_num_batched_tokens >= max_num_seqs.
        return [cfg.model_copy(update={"prefill_budget": b}) for b in self.PREFILL_BUDGETS
                if b >= cfg.batch and b != cfg.prefill_budget]

    def disagg_launch_spec(self, cfg: Config, model: PreparedModel, port: int, role: str,
                           kv_transfer_config: dict, gpu_index: int, side_channel_port: int) -> LaunchSpec:
        spec = self.launch_spec(cfg, model, port)
        spec.args += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
        spec.env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
        })
        return spec

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/health",
            model_name=model.spec.hf_id,  # --served-model-name
            tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if hw.gpu else [],
        )
