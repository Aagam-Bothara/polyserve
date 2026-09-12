"""Core data types shared across the pipeline.

Everything here is plain data (pydantic) so it can be serialised into the
profile cache and printed by the CLI without importing any backend.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

Objective = Literal["throughput", "latency", "balanced", "efficiency"]
OBJECTIVES: Tuple[str, ...] = ("throughput", "latency", "balanced", "efficiency")

MiB = 1024 * 1024
GiB = 1024 * MiB


# --------------------------------------------------------------------------- hardware


class GPUInfo(BaseModel):
    vendor: str  # "nvidia" | "amd" | "apple" | "unknown"
    name: str
    index: int = 0
    compute_capability: Optional[Tuple[int, int]] = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    uuid: Optional[str] = None

    @property
    def cc(self) -> Tuple[int, int]:
        return self.compute_capability or (0, 0)


class CPUInfo(BaseModel):
    model_name: str = "unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    avx2: bool = False
    avx512: bool = False
    ram_total_bytes: int = 0
    ram_free_bytes: int = 0
    arch: str = "unknown"  # x86_64 | aarch64 | ...


class BackendAvailability(BaseModel):
    name: str
    available: bool
    version: Optional[str] = None
    reason: Optional[str] = None


class HardwareDescriptor(BaseModel):
    os: str
    python: str
    gpus: List[GPUInfo] = Field(default_factory=list)
    cpu: CPUInfo
    backends: Dict[str, BackendAvailability] = Field(default_factory=dict)
    probed_at: float = Field(default_factory=time.time)

    @property
    def gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def backend_available(self, name: str) -> bool:
        b = self.backends.get(name)
        return bool(b and b.available)

    def backend_version(self, name: str) -> Optional[str]:
        b = self.backends.get(name)
        return b.version if b else None

    @property
    def available_memory_bytes(self) -> int:
        """Memory the planner budgets against: free VRAM if a GPU exists, else free RAM."""
        if self.gpu is not None:
            return self.gpu.vram_free_bytes
        return self.cpu.ram_free_bytes


# --------------------------------------------------------------------------- model


class ModelSpec(BaseModel):
    hf_id: str
    revision: Optional[str] = None

    @property
    def safe_id(self) -> str:
        return self.hf_id.replace("/", "__")


class ArchInfo(BaseModel):
    """What the memory planner needs from an HF config."""

    architecture: str = "unknown"
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    num_params: Optional[int] = None
    torch_dtype: str = "bfloat16"

    def kv_bytes_per_token(self, kv_dtype_bytes: int) -> int:
        # 2 (K and V) x layers x kv_heads x head_dim x bytes.  For MHA models
        # kv_heads*head_dim == hidden_size, which matches the spec formula; for GQA
        # models this is the correct (smaller) number.
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * kv_dtype_bytes


class PreparedModel(BaseModel):
    spec: ModelSpec
    backend: str
    arch: ArchInfo
    # vLLM/SGLang: the HF id (or local snapshot dir). llama.cpp: path to a GGUF file per quant.
    hf_path: Optional[str] = None
    gguf_paths: Dict[str, str] = Field(default_factory=dict)  # quant -> path
    weights_bytes: Dict[str, int] = Field(default_factory=dict)  # quant/precision -> bytes

    def weights_for(self, quant: str) -> int:
        if quant in self.weights_bytes:
            return self.weights_bytes[quant]
        raise KeyError(f"no weight size recorded for quant {quant!r}")


# --------------------------------------------------------------------------- config


class Config(BaseModel):
    """One concrete launch configuration for a backend.

    Fields are the union of what all v1 backends understand; each backend
    ignores the ones that do not apply to it.
    """

    backend: str
    quant: str  # "bf16" | "fp16" | "fp8" | "Q4_K_M" | ...
    ctx: int = 4096  # max model length / context window
    batch: int = 64  # max_num_seqs (vLLM/SGLang) or n_parallel (llama.cpp)
    gpu_memory_utilization: Optional[float] = None  # vLLM/SGLang
    n_gpu_layers: Optional[int] = None  # llama.cpp
    n_batch: Optional[int] = None  # llama.cpp logical batch
    kv_dtype: str = "auto"
    extra: Dict[str, Any] = Field(default_factory=dict)
    # Energy tuning, applied through NVML while the backend runs rather than as launch flags.
    power_limit_w: Optional[int] = None  # board power cap
    sm_clock_mhz: Optional[int] = None  # upper bound of the locked SM clock range

    def key(self) -> str:
        parts = [self.backend, self.quant, f"ctx{self.ctx}", f"b{self.batch}"]
        if self.gpu_memory_utilization is not None:
            parts.append(f"gmu{self.gpu_memory_utilization:.2f}")
        if self.n_gpu_layers is not None:
            parts.append(f"ngl{self.n_gpu_layers}")
        if self.n_batch is not None:
            parts.append(f"nb{self.n_batch}")
        if self.kv_dtype != "auto":
            parts.append(f"kv{self.kv_dtype}")
        if self.power_limit_w is not None:
            parts.append(f"pl{self.power_limit_w}")
        if self.sm_clock_mhz is not None:
            parts.append(f"clk{self.sm_clock_mhz}")
        return "/".join(parts)

    def base_key(self) -> str:
        """Key without energy settings, so power variants of one configuration share it."""
        return self.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}).key()

    def short(self) -> str:
        return self.key()


class MemoryEstimate(BaseModel):
    config_key: str
    weights: int
    kv_cache: int
    runtime_workspace: int
    safety_margin: int
    total: int
    budget: int
    feasible: bool

    @property
    def total_gib(self) -> float:
        return self.total / GiB


# --------------------------------------------------------------------------- calibration


class TrialMetrics(BaseModel):
    tok_s: float = 0.0  # aggregate output tokens per second
    ttft_ms: Optional[float] = None  # p50 time to first token (None = not measured / failed)
    ttft_p95_ms: Optional[float] = None
    tpot_ms: Optional[float] = None  # mean time per output token after the first
    peak_mem_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    power_w: Optional[float] = None  # mean device power during the trial
    sm_clock_mhz: Optional[float] = None  # mean observed SM clock; confirms a clock lock took effect
    joules_per_token: Optional[float] = None
    duration_s: float = 0.0
    requests: int = 0
    failed: int = 0
    output_tokens: int = 0
    concurrency: int = 1
    telemetry_source: str = "none"  # "llmtrace" | "pynvml" | "psutil" | "none"
    token_count_source: str = "none"  # "usage" | "tokenizer" | "chunks" (approximate) | "none"
    prompt_tokens: int = 0  # mean measured prompt length (0 if no tokenizer)
    by_concurrency: Dict[str, "TrialMetrics"] = Field(default_factory=dict)

    # A trial is not invalidated by a couple of degenerate requests. Small models sometimes emit
    # end-of-sequence immediately, producing no tokens; a backend that is actually broken fails
    # everything, not 2 requests in 48.
    FAILURE_TOLERANCE: ClassVar[float] = 0.10

    @property
    def ok(self) -> bool:
        if self.requests <= 0 or self.output_tokens <= 0:
            return False
        return self.failed <= self.FAILURE_TOLERANCE * self.requests


class MemoryObservation(BaseModel):
    """Planner prediction next to what the backend actually allocated, for one trial."""

    predicted: Optional[MemoryEstimate] = None
    measured: Optional["MeasuredMemory"] = None


class TrialResult(BaseModel):
    config: Config
    stage: str
    metrics: TrialMetrics
    launched: bool = True
    error: Optional[str] = None
    started_at: float = Field(default_factory=time.time)
    memory: Optional[MemoryObservation] = None

    @property
    def ok(self) -> bool:
        return self.launched and self.error is None and self.metrics.ok


from polyserve.memlog import MeasuredMemory  # noqa: E402  (after MemoryEstimate is defined)

MemoryObservation.model_rebuild()


class Profile(BaseModel):
    """What gets cached: the winner plus everything needed to reproduce the decision."""

    polyserve_version: str
    hardware_hash: str
    hardware: HardwareDescriptor
    model_id: str
    objective: str
    workload: str = "default"
    workload_spec: Dict[str, Any] = Field(default_factory=dict)
    power_mode: str = "off"  # "off" | "cap" | "clock" | "both"
    backend: str
    backend_version: Optional[str]
    config: Config
    prepared: Optional[PreparedModel] = None
    # Every candidate backend's prepared model, so trials from losing backends stay analysable.
    prepared_all: Dict[str, PreparedModel] = Field(default_factory=dict)
    launch_args: List[str]
    launch_env: Dict[str, str] = Field(default_factory=dict)
    calibration_table: List[TrialResult] = Field(default_factory=list)
    calibration_seconds: float = 0.0
    calibration_trials: int = 0
    llmtrace_version: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    notes: List[str] = Field(default_factory=list)
