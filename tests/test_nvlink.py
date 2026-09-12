"""NVLink detection, which decides how tensor-parallel engines are launched."""

from __future__ import annotations

import sys
import types

from polyserve.hardware import nvlink_between


class _NotSupported(Exception):
    pass


def _fake_nvml(links):
    """links: GPU index -> list of per-link states (1 = enabled), or None when NVLink is unsupported."""

    def state(handle, link):
        if links[handle] is None or link >= len(links[handle]):
            raise _NotSupported()
        return links[handle][link]

    return types.SimpleNamespace(NVML_FEATURE_ENABLED=1, nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                                 nvmlDeviceGetHandleByIndex=lambda i: i, nvmlDeviceGetNvLinkState=state)


def test_nvlink_detection(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml({0: [0, 1], 1: [1]}))
    assert nvlink_between([0, 1]) is True  # an inactive first link does not hide an active second one
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml({0: None, 1: None}))
    assert nvlink_between([0, 1]) is False  # PCIe only
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml({0: [1], 1: None}))
    assert nvlink_between([0, 1]) is False  # one card without NVLink is enough to lose it
    monkeypatch.setitem(sys.modules, "pynvml", None)  # not installed
    assert nvlink_between([0, 1]) is None


def test_tensor_parallel_shapes_leave_room_for_nccl():
    from polyserve.layout import TP_MAX_GPU_MEMORY_UTILIZATION, tp_candidates
    from polyserve.models import Config

    shapes = tp_candidates(Config(backend="vllm", quant="awq", batch=256, gpu_memory_utilization=0.95), 2)
    assert [c.batch for c in shapes] == [256, 512] and all(c.tp == 2 for c in shapes)
    assert all(c.gpu_memory_utilization == TP_MAX_GPU_MEMORY_UTILIZATION for c in shapes)
    low = tp_candidates(Config(backend="vllm", quant="awq", batch=64, gpu_memory_utilization=0.8), 2)
    assert all(c.gpu_memory_utilization == 0.8 for c in low)  # never raised


def test_tensor_parallel_over_pcie_routes_nccl_through_the_host(prepared_vllm, monkeypatch):
    import polyserve.backends.sglang as sg
    import polyserve.backends.vllm as vl
    from polyserve.backends import get_backend
    from polyserve.models import Config

    for mod, name in ((vl, "vllm"), (sg, "sglang")):
        cfg = Config(backend=name, quant="bf16", tp=2)
        monkeypatch.setattr(mod, "nvlink_between", lambda indices: False)
        pcie = get_backend(name).launch_spec(cfg, prepared_vllm, 1)
        assert pcie.env.get("NCCL_P2P_DISABLE") == "1" and "--disable-custom-all-reduce" in pcie.args
        monkeypatch.setattr(mod, "nvlink_between", lambda indices: True)  # NVLink: leave NCCL alone
        nvlink = get_backend(name).launch_spec(cfg, prepared_vllm, 1)
        assert "NCCL_P2P_DISABLE" not in nvlink.env and "--disable-custom-all-reduce" not in nvlink.args
        monkeypatch.setattr(mod, "nvlink_between", lambda indices: None)  # unknown: leave NCCL alone
        assert "NCCL_P2P_DISABLE" not in get_backend(name).launch_spec(cfg, prepared_vllm, 1).env
        single = Config(backend=name, quant="bf16")
        monkeypatch.setattr(mod, "nvlink_between", lambda indices: False)
        assert "NCCL_P2P_DISABLE" not in get_backend(name).launch_spec(single, prepared_vllm, 1).env
