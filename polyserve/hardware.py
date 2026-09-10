"""Step 1: probe. Produce a HardwareDescriptor and a stable hardware hash."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import psutil

from polyserve.models import BackendAvailability, CPUInfo, GPUInfo, HardwareDescriptor

logger = logging.getLogger(__name__)

BACKEND_NAMES = ("vllm", "sglang", "llamacpp-cuda", "llamacpp-cpu", "vllm-cpu")


# --------------------------------------------------------------------------- GPU


def _probe_gpus_pynvml() -> List[GPUInfo]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # NVML present but no driver
        logger.debug("nvmlInit failed: %s", exc)
        return []
    gpus: List[GPUInfo] = []
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        try:
            cuda_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_version: Optional[str] = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        except Exception:
            cuda_version = None
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc: Optional[Tuple[int, int]] = (int(major), int(minor))
            except Exception:
                cc = None
            try:
                uuid = pynvml.nvmlDeviceGetUUID(h)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
            except Exception:
                uuid = None
            gpus.append(
                GPUInfo(
                    vendor="nvidia",
                    name=str(name),
                    index=i,
                    compute_capability=cc,
                    vram_total_bytes=int(mem.total),
                    vram_free_bytes=int(mem.free),
                    driver_version=str(driver),
                    cuda_version=cuda_version,
                    uuid=uuid,
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def _probe_gpus_torch() -> List[GPUInfo]:
    if importlib.util.find_spec("torch") is None:
        return []
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append(
                GPUInfo(
                    vendor="nvidia" if not getattr(torch.version, "hip", None) else "amd",
                    name=props.name,
                    index=i,
                    compute_capability=(props.major, props.minor),
                    vram_total_bytes=int(total),
                    vram_free_bytes=int(free),
                    cuda_version=getattr(torch.version, "cuda", None),
                )
            )
        return gpus
    except Exception as exc:
        logger.debug("torch GPU probe failed: %s", exc)
        return []


def _probe_gpus_nvidia_smi() -> List[GPUInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, free, driver, cc = parts[:6]
        try:
            ccm = tuple(int(x) for x in cc.split("."))
            cct: Optional[Tuple[int, int]] = (ccm[0], ccm[1])
        except Exception:
            cct = None
        gpus.append(
            GPUInfo(
                vendor="nvidia",
                name=name,
                index=int(idx),
                compute_capability=cct,
                vram_total_bytes=int(float(total)) * 1024 * 1024,
                vram_free_bytes=int(float(free)) * 1024 * 1024,
                driver_version=driver,
            )
        )
    return gpus


def probe_gpus() -> List[GPUInfo]:
    for fn in (_probe_gpus_pynvml, _probe_gpus_torch, _probe_gpus_nvidia_smi):
        gpus = fn()
        if gpus:
            return gpus
    return []


# --------------------------------------------------------------------------- CPU


def _cpu_flags() -> Tuple[str, set]:
    """Return (model name, flag set) from /proc/cpuinfo, py-cpuinfo, or nothing."""
    name = platform.processor() or "unknown"
    flags: set = set()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            if m:
                name = m.group(1).strip()
            m = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
            if m:
                flags = set(m.group(1).split())
            return name, flags
        except OSError:
            pass
    try:  # optional dependency, helps on macOS/Windows dev boxes
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        name = info.get("brand_raw", name)
        flags = set(info.get("flags", []))
    except Exception:
        pass
    return name, flags


def probe_cpu() -> CPUInfo:
    name, flags = _cpu_flags()
    vm = psutil.virtual_memory()
    return CPUInfo(
        model_name=name,
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        avx2="avx2" in flags,
        avx512=any(f.startswith("avx512") for f in flags),
        ram_total_bytes=int(vm.total),
        ram_free_bytes=int(vm.available),
        arch=platform.machine() or "unknown",
    )


# --------------------------------------------------------------------------- backends


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def llama_server_binary() -> Optional[str]:
    """Path to llama-server; honours $LLAMA_SERVER."""
    env = os.environ.get("LLAMA_SERVER")
    if env and os.path.exists(env):
        return env
    for cand in ("llama-server", "llama-server.exe"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def _llama_server_info(binary: str) -> Tuple[Optional[str], bool]:
    """Return (version string, has_cuda) by invoking llama-server."""
    version: Optional[str] = None
    has_cuda = False
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        text = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"version:\s*(\S+)", text)
        if m:
            version = m.group(1)
        if re.search(r"\bCUDA\b", text, re.I):
            has_cuda = True
    except Exception as exc:
        logger.debug("llama-server --version failed: %s", exc)
    if not has_cuda:
        try:
            out = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=15)
            text = (out.stdout or "") + (out.stderr or "")
            if re.search(r"\bCUDA\d*\b", text):
                has_cuda = True
        except Exception:
            pass
    return version, has_cuda


def _torch_is_cuda_build() -> Optional[bool]:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch  # type: ignore

        return bool(getattr(torch.version, "cuda", None))
    except Exception:
        return None


def probe_backends(gpus: List[GPUInfo]) -> Dict[str, BackendAvailability]:
    out: Dict[str, BackendAvailability] = {}
    has_nvidia = any(g.vendor == "nvidia" for g in gpus)

    vllm_ver = _pkg_version("vllm")
    vllm_importable = importlib.util.find_spec("vllm") is not None
    torch_cuda = _torch_is_cuda_build()
    out["vllm"] = BackendAvailability(
        name="vllm",
        available=bool(vllm_importable and has_nvidia and torch_cuda is not False),
        version=vllm_ver,
        reason=None if vllm_importable else "vllm not importable",
    )
    # vLLM-CPU: the same package built against a CPU-only torch.
    out["vllm-cpu"] = BackendAvailability(
        name="vllm-cpu",
        available=bool(vllm_importable and torch_cuda is False),
        version=vllm_ver,
        reason=(
            "vllm not importable" if not vllm_importable
            else ("torch is a CUDA build; vLLM-CPU needs the CPU wheel" if torch_cuda else None)
        ),
    )

    sgl_ver = _pkg_version("sglang")
    sgl_importable = importlib.util.find_spec("sglang") is not None
    out["sglang"] = BackendAvailability(
        name="sglang",
        available=bool(sgl_importable and has_nvidia),
        version=sgl_ver,
        reason=None if sgl_importable else "sglang not importable",
    )

    binary = llama_server_binary()
    if binary:
        ver, has_cuda = _llama_server_info(binary)
        out["llamacpp-cuda"] = BackendAvailability(
            name="llamacpp-cuda",
            available=bool(has_cuda and has_nvidia),
            version=ver,
            reason=None if has_cuda else "llama-server built without CUDA",
        )
        out["llamacpp-cpu"] = BackendAvailability(name="llamacpp-cpu", available=True, version=ver)
    else:
        for n in ("llamacpp-cuda", "llamacpp-cpu"):
            out[n] = BackendAvailability(
                name=n, available=False, reason="llama-server not on PATH (set $LLAMA_SERVER)"
            )
    return out


# --------------------------------------------------------------------------- entry


def probe() -> HardwareDescriptor:
    gpus = probe_gpus()
    cpu = probe_cpu()
    return HardwareDescriptor(
        os=f"{platform.system()} {platform.release()}",
        python=sys.version.split()[0],
        gpus=gpus,
        cpu=cpu,
        backends=probe_backends(gpus),
    )


def hardware_hash(hw: HardwareDescriptor) -> str:
    """Stable identity of the machine for cache keys.

    Uses the *shape* of the hardware (GPU model, CC, total VRAM, CPU model,
    cores, SIMD flags, total RAM), not volatile fields like free memory.
    """
    parts = []
    for g in hw.gpus:
        parts.append(f"gpu:{g.vendor}:{g.name}:{g.cc[0]}.{g.cc[1]}:{g.vram_total_bytes // (256 * 1024 * 1024)}")
    c = hw.cpu
    parts.append(f"cpu:{c.model_name}:{c.physical_cores}:{c.logical_cores}:{int(c.avx2)}{int(c.avx512)}")
    parts.append(f"ram:{c.ram_total_bytes // (1024 * 1024 * 1024)}")
    parts.append(f"arch:{c.arch}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def llmtrace_version() -> Optional[str]:
    return _pkg_version("llmtrace")
