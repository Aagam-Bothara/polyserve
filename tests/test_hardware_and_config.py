from __future__ import annotations

from polyserve.hardware import hardware_hash, probe
from polyserve.hfconfig import arch_from_config, dtype_bytes
from tests.conftest import LLAMA_3B_CONFIG, make_hw


def test_hardware_hash_is_stable_and_ignores_free_memory():
    a = make_hw("a100")
    b = make_hw("a100")
    b.gpus[0].vram_free_bytes //= 2
    b.cpu.ram_free_bytes //= 3
    assert hardware_hash(a) == hardware_hash(b)
    assert len(hardware_hash(a)) == 16


def test_hardware_hash_differs_across_machines():
    hashes = {hardware_hash(make_hw(k)) for k in ("a100", "rtx4090", "gtx1080", "cpu")}
    assert len(hashes) == 4


def test_probe_runs_on_this_machine():
    hw = probe()
    assert hw.cpu.logical_cores >= 1
    assert set(hw.backends) == {"vllm", "sglang", "llamacpp-cuda", "llamacpp-cpu", "vllm-cpu"}


def test_arch_from_llama_config():
    arch = arch_from_config(LLAMA_3B_CONFIG)
    assert arch.architecture == "LlamaForCausalLM"
    assert arch.num_layers == 28 and arch.num_kv_heads == 8 and arch.head_dim == 128
    # Llama-3.2-3B has 3.21B parameters; the estimate should land within 3%.
    assert abs(arch.num_params - 3.21e9) / 3.21e9 < 0.03
    # KV bytes/token in bf16: 2 * 28 * 8 * 128 * 2 = 114688
    assert arch.kv_bytes_per_token(2) == 114_688


def test_arch_nested_text_config():
    cfg = {"architectures": ["Gemma3ForConditionalGeneration"], "text_config": dict(LLAMA_3B_CONFIG)}
    arch = arch_from_config(cfg)
    assert arch.num_layers == 28


def test_dtype_bytes():
    assert dtype_bytes("bfloat16") == 2 and dtype_bytes("fp8") == 1 and dtype_bytes("float32") == 4


def test_cuda_visible_devices_is_honoured(monkeypatch):
    """NVML enumerates physical devices regardless of CUDA_VISIBLE_DEVICES; the backends do not."""
    from polyserve.hardware import visible_gpus
    from polyserve.models import GPUInfo

    gpus = [GPUInfo(vendor="nvidia", name="A", index=0, uuid="GPU-aaa"),
            GPUInfo(vendor="nvidia", name="B", index=1, uuid="GPU-bbb")]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_gpus(gpus) == gpus
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert visible_gpus(gpus) == []          # CPU-only as far as any backend is concerned
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert visible_gpus(gpus) == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert [g.name for g in visible_gpus(gpus)] == ["B"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa")
    assert [g.name for g in visible_gpus(gpus)] == ["A"]


def test_backend_reasons_explain_a_masked_gpu(monkeypatch):
    """With no visible GPU an installed vLLM must say why, not report an empty reason."""
    from polyserve.hardware import probe_backends

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name in ("vllm", "sglang") else None)
    monkeypatch.setattr("polyserve.hardware._torch_is_cuda_build", lambda: True)
    out = probe_backends([])  # no visible GPUs
    assert not out["vllm"].available and "NVIDIA" in out["vllm"].reason
    assert not out["sglang"].available and "NVIDIA" in out["sglang"].reason
