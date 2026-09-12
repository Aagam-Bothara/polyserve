from __future__ import annotations

import os
from typing import Dict

import pytest

from polyserve.models import (
    ArchInfo,
    BackendAvailability,
    CPUInfo,
    GiB,
    GPUInfo,
    HardwareDescriptor,
    ModelSpec,
    PreparedModel,
)

# Llama-3.2-3B-Instruct config.json (the numbers that matter for planning).
LLAMA_3B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
}


def _backends(**avail: bool) -> Dict[str, BackendAvailability]:
    names = ("vllm", "sglang", "llamacpp-cuda", "llamacpp-cpu", "vllm-cpu")
    out = {}
    for n in names:
        a = avail.get(n.replace("-", "_"), False)
        out[n] = BackendAvailability(name=n, available=a, version=("1.0" if a else None))
    return out


def make_hw(kind: str) -> HardwareDescriptor:
    cpu = CPUInfo(
        model_name="Test CPU", physical_cores=8, logical_cores=16, avx2=True, avx512=(kind == "cpu-avx512"),
        ram_total_bytes=64 * GiB, ram_free_bytes=48 * GiB, arch="x86_64",
    )
    if kind == "a100":
        gpus = [GPUInfo(vendor="nvidia", name="NVIDIA A100-SXM4-80GB", compute_capability=(8, 0),
                        vram_total_bytes=80 * GiB, vram_free_bytes=79 * GiB)]
        backends = _backends(vllm=True, sglang=True, llamacpp_cuda=True, llamacpp_cpu=True)
    elif kind == "rtx4090":
        gpus = [GPUInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4090", compute_capability=(8, 9),
                        vram_total_bytes=24 * GiB, vram_free_bytes=23 * GiB)]
        backends = _backends(vllm=True, sglang=True, llamacpp_cuda=True, llamacpp_cpu=True)
    elif kind == "gtx1080":
        gpus = [GPUInfo(vendor="nvidia", name="NVIDIA GeForce GTX 1080", compute_capability=(6, 1),
                        vram_total_bytes=8 * GiB, vram_free_bytes=7 * GiB)]
        backends = _backends(vllm=True, llamacpp_cuda=True, llamacpp_cpu=True)  # vllm installed but unusable
    elif kind in ("cpu", "cpu-avx512"):
        gpus = []
        backends = _backends(llamacpp_cpu=True, vllm_cpu=(kind == "cpu-avx512"))
    else:
        raise ValueError(kind)
    return HardwareDescriptor(os="Linux test", python="3.12", gpus=gpus, cpu=cpu, backends=backends)


@pytest.fixture
def hw_a100():
    return make_hw("a100")


@pytest.fixture
def hw_rtx4090():
    return make_hw("rtx4090")


@pytest.fixture
def hw_gtx1080():
    return make_hw("gtx1080")


@pytest.fixture
def hw_cpu():
    return make_hw("cpu")


@pytest.fixture
def hw_cpu_avx512():
    return make_hw("cpu-avx512")


@pytest.fixture
def llama3b_arch() -> ArchInfo:
    from polyserve.hfconfig import arch_from_config

    return arch_from_config(LLAMA_3B_CONFIG)


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(hf_id="meta-llama/Llama-3.2-3B-Instruct")


@pytest.fixture
def prepared_vllm(spec, llama3b_arch) -> PreparedModel:
    p = llama3b_arch.num_params
    return PreparedModel(spec=spec, backend="vllm", arch=llama3b_arch, hf_path=spec.hf_id,
                         weights_bytes={"bf16": p * 2, "fp8": p})


@pytest.fixture
def prepared_llamacpp(spec, llama3b_arch) -> PreparedModel:
    from polyserve.gguf import GGUF_QUANTS, estimate_gguf_bytes

    p = llama3b_arch.num_params
    return PreparedModel(
        spec=spec, backend="llamacpp-cuda", arch=llama3b_arch,
        gguf_paths={q: f"/models/llama-{q}.gguf" for q in GGUF_QUANTS},
        weights_bytes={q: estimate_gguf_bytes(p, q) for q in GGUF_QUANTS},
    )


@pytest.fixture
def no_network(monkeypatch, llama3b_arch):
    """Backends read HF configs via load_arch; pin it so tests never touch the hub."""
    import polyserve.backends.llamacpp as lc
    import polyserve.backends.sglang as sg
    import polyserve.backends.vllm as vl
    import polyserve.backends.vllm_cpu as vc

    for mod in (lc, sg, vl, vc):
        monkeypatch.setattr(mod, "load_arch", lambda spec, token=None: llama3b_arch)
    monkeypatch.setattr(vl, "vllm_registry_archs", lambda: None)
    monkeypatch.setattr(sg, "sglang_registry_archs", lambda: None)
    monkeypatch.setattr(lc, "search_hub_gguf", lambda spec, quants, **kw: {})
    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/usr/local/bin/llama-server")
    import polyserve.quantized as qz

    monkeypatch.setattr(qz, "find_int4_repos", lambda spec, **kw: {})
    yield


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYSERVE_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


@pytest.fixture(autouse=True)
def _no_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
