"""Extract measured memory components from backend logs.

The planner predicts weights + kv + workspace. To calibrate it we need the same components as the
backend actually allocated. vLLM, SGLang and llama.cpp all print them at startup; NVML peak
(measured by the trial telemetry) is the ground truth for the total.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

MiB = 1024 * 1024
GiB = 1024 * MiB


class MeasuredMemory(BaseModel):
    weights_mb: Optional[float] = None
    kv_mb: Optional[float] = None  # KV cache the backend actually allocated
    workspace_mb: Optional[float] = None  # everything else on the device (context, graphs, activations)
    non_kv_mb: Optional[float] = None  # weights + workspace as the backend reports it
    kv_tokens: Optional[int] = None  # tokens the allocated KV cache can hold
    device_peak_mb: Optional[float] = None  # NVML peak during the trial, minus what was used before launch
    baseline_mb: Optional[float] = None  # device memory used before launch
    source: str = "none"  # "vllm-log" | "llamacpp-log" | "sglang-log" | "nvml" | "psutil"
    details: Dict[str, float] = Field(default_factory=dict)

    @property
    def total_mb(self) -> Optional[float]:
        """Best available total: device peak, else the log's components."""
        if self.device_peak_mb is not None:
            return self.device_peak_mb
        parts = [p for p in (self.weights_mb, self.kv_mb, self.workspace_mb) if p is not None]
        return sum(parts) if parts else None


_SIZE = r"([\d,.]+)\s*(GiB|MiB|GB|MB|KiB|KB)"


def _to_mb(num: str, unit: str) -> float:
    v = float(num.replace(",", ""))
    unit = unit.lower()
    if unit in ("gib", "gb"):
        return v * 1024
    if unit in ("kib", "kb"):
        return v / 1024
    return v


def parse_vllm(text: str) -> MeasuredMemory:
    """vLLM 0.10/0.11 startup lines.

    Examples:
      Model loading took 6.4265 GiB memory and 8.16 seconds
      Total non KV cache memory: 7.85GiB; torch peak memory increase: 1.0GiB; non-torch forward increase
        memory: 0.1GiB; weights memory: 6.4GiB.
      Available KV cache memory: 12.34 GiB
      GPU KV cache size: 353,296 tokens
    """
    m = MeasuredMemory(source="none")
    d: Dict[str, float] = {}
    if r := re.search(r"weights memory:\s*" + _SIZE, text):
        m.weights_mb = _to_mb(*r.groups())
    elif r := re.search(r"Model loading took\s*" + _SIZE, text):
        # vLLM 0.11: "Model loading took 5.7916 GiB and 1.16 seconds" (no "memory" word).
        m.weights_mb = _to_mb(*r.groups())
    if r := re.search(r"Total non KV cache memory:\s*" + _SIZE, text):
        m.non_kv_mb = _to_mb(*r.groups())
    if r := re.search(r"torch peak memory increase:\s*" + _SIZE, text):
        d["torch_peak_increase_mb"] = _to_mb(*r.groups())
    if r := re.search(r"non-torch forward increase memory:\s*" + _SIZE, text):
        d["non_torch_increase_mb"] = _to_mb(*r.groups())
    if r := re.search(r"Available KV cache memory:\s*" + _SIZE, text):
        m.kv_mb = _to_mb(*r.groups())
    if r := re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text):
        m.kv_tokens = int(r.group(1).replace(",", ""))
    if m.non_kv_mb is not None and m.weights_mb is not None:
        m.workspace_mb = max(0.0, m.non_kv_mb - m.weights_mb)
    if any(v is not None for v in (m.weights_mb, m.kv_mb, m.non_kv_mb)):
        m.source = "vllm-log"
    m.details = d
    return m


def parse_llamacpp(text: str) -> MeasuredMemory:
    """llama-server startup lines (device buffers only; CPU_Mapped / host buffers are recorded separately).

    Examples:
      load_tensors: CUDA0 model buffer size =  2003.50 MiB
      load_tensors:   CPU_Mapped model buffer size =   308.23 MiB
      llama_kv_cache_unified: CUDA0 KV buffer size =   576.00 MiB
      llama_context:      CUDA0 compute buffer size =   300.25 MiB
      llama_context:  CUDA_Host compute buffer size =    16.01 MiB
    """
    m = MeasuredMemory(source="none")
    d: Dict[str, float] = {}
    dev_w = host_w = dev_kv = host_kv = dev_c = host_c = 0.0
    found = False
    for r in re.finditer(r"(\S+)\s+model buffer size\s*=\s*" + _SIZE, text):
        found = True
        dev, num, unit = r.groups()
        if dev.upper().startswith(("CUDA", "ROCM", "METAL", "VULKAN", "SYCL")) and "HOST" not in dev.upper():
            dev_w += _to_mb(num, unit)
        else:
            host_w += _to_mb(num, unit)
    for r in re.finditer(r"(\S+)\s+KV buffer size\s*=\s*" + _SIZE, text):
        found = True
        dev, num, unit = r.groups()
        if dev.upper().startswith(("CUDA", "ROCM", "METAL", "VULKAN", "SYCL")) and "HOST" not in dev.upper():
            dev_kv += _to_mb(num, unit)
        else:
            host_kv += _to_mb(num, unit)
    for r in re.finditer(r"(\S+)\s+compute buffer size\s*=\s*" + _SIZE, text):
        found = True
        dev, num, unit = r.groups()
        if dev.upper().startswith(("CUDA", "ROCM", "METAL", "VULKAN", "SYCL")) and "HOST" not in dev.upper():
            dev_c += _to_mb(num, unit)
        else:
            host_c += _to_mb(num, unit)
    if r := re.search(r"n_ctx\s*=\s*(\d+)", text):
        m.kv_tokens = int(r.group(1))
    if not found:
        return m
    gpu = "CUDA" in text.upper() and dev_w + dev_kv + dev_c > 0
    if gpu:
        m.weights_mb, m.kv_mb, m.workspace_mb = dev_w, dev_kv, dev_c
        d.update(host_weights_mb=host_w, host_kv_mb=host_kv, host_compute_mb=host_c)
    else:
        m.weights_mb, m.kv_mb, m.workspace_mb = host_w, host_kv, host_c
    m.non_kv_mb = (m.weights_mb or 0) + (m.workspace_mb or 0)
    m.source = "llamacpp-log"
    m.details = d
    return m


def parse_sglang(text: str) -> MeasuredMemory:
    """SGLang startup lines.

    Examples:
      Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=15.20 GB, mem usage=6.21 GB.
      KV Cache is allocated. #tokens: 123456, K size: 2.12 GB, V size: 2.12 GB
      Memory pool end. avail mem=8.10 GB
    """
    m = MeasuredMemory(source="none")
    if r := re.search(r"Load weight end\..*?mem usage=" + _SIZE, text):
        m.weights_mb = _to_mb(*r.groups())
    ks = vs = None
    if r := re.search(r"K size:\s*" + _SIZE, text):
        ks = _to_mb(*r.groups())
    if r := re.search(r"V size:\s*" + _SIZE, text):
        vs = _to_mb(*r.groups())
    if ks is not None or vs is not None:
        m.kv_mb = (ks or 0) + (vs or 0)
    if r := re.search(r"#tokens:\s*([\d,]+)", text):
        m.kv_tokens = int(r.group(1).replace(",", ""))
    if m.weights_mb is not None or m.kv_mb is not None:
        m.source = "sglang-log"
    return m


PARSERS = {
    "vllm": parse_vllm,
    "vllm-cpu": parse_vllm,
    "sglang": parse_sglang,
    "llamacpp-cuda": parse_llamacpp,
    "llamacpp-cpu": parse_llamacpp,
}


def parse_log(backend: str, text: str) -> MeasuredMemory:
    fn = PARSERS.get(backend)
    return fn(text) if fn else MeasuredMemory()


def merge(log: MeasuredMemory, device_peak_mb: Optional[float], baseline_mb: Optional[float],
          telemetry_source: str) -> MeasuredMemory:
    """Combine log components with the NVML/psutil peak observed during the trial.

    vLLM 0.11 reports weights and the KV pool but not the rest, so workspace is derived as
    peak - weights - kv when all three are known. That residual is exactly the non-weight,
    non-KV device memory (CUDA context, graphs, activations) the planner budgets for.
    """
    out = log.model_copy()
    if device_peak_mb is not None:
        peak = device_peak_mb - (baseline_mb or 0.0)
        out.device_peak_mb = max(0.0, peak)
        out.baseline_mb = baseline_mb
        if out.source == "none":
            out.source = "nvml" if telemetry_source in ("llmtrace", "pynvml") else telemetry_source
    if out.workspace_mb is None and out.device_peak_mb and out.weights_mb is not None and out.kv_mb is not None:
        out.workspace_mb = max(0.0, out.device_peak_mb - out.weights_mb - out.kv_mb)
        out.details["workspace_derived"] = 1.0
    if out.non_kv_mb is None and out.weights_mb is not None and out.workspace_mb is not None:
        out.non_kv_mb = out.weights_mb + out.workspace_mb
    return out


def device_used_mb(gpu_ids: List[int]) -> Optional[float]:
    """Device memory in use right now (before launching a backend), via NVML."""
    if not gpu_ids:
        return None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            return sum(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used for i in gpu_ids) / MiB
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
