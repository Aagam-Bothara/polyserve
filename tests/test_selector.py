from __future__ import annotations

import pytest

from polyserve.backends import registry
from polyserve.selector import select_backends


@pytest.mark.usefixtures("no_network")
def test_modern_gpu_gets_vllm_sglang_and_llamacpp(hw_a100, spec):
    assert select_backends(hw_a100, spec, registry()) == ["vllm", "sglang", "llamacpp-cuda"]


@pytest.mark.usefixtures("no_network")
def test_pascal_only_gets_llamacpp_cuda(hw_gtx1080, spec):
    # vllm is "installed" on this box but cc 6.1 < 7.5 so it must not be selected.
    assert select_backends(hw_gtx1080, spec, registry()) == ["llamacpp-cuda"]


@pytest.mark.usefixtures("no_network")
def test_cpu_without_avx512(hw_cpu, spec):
    assert select_backends(hw_cpu, spec, registry()) == ["llamacpp-cpu"]


@pytest.mark.usefixtures("no_network")
def test_cpu_with_avx512_adds_vllm_cpu(hw_cpu_avx512, spec):
    assert select_backends(hw_cpu_avx512, spec, registry()) == ["llamacpp-cpu", "vllm-cpu"]


@pytest.mark.usefixtures("no_network")
def test_force_backend(hw_cpu, spec):
    assert select_backends(hw_cpu, spec, registry(), force="vllm") == ["vllm"]
    with pytest.raises(ValueError):
        select_backends(hw_cpu, spec, registry(), force="mlx")


@pytest.mark.usefixtures("no_network")
def test_unavailable_backend_excluded(hw_a100, spec):
    hw_a100.backends["sglang"].available = False
    assert "sglang" not in select_backends(hw_a100, spec, registry())


@pytest.mark.usefixtures("no_network")
def test_unsupported_arch_excluded_from_vllm(hw_a100, spec, monkeypatch, llama3b_arch):
    import polyserve.backends.vllm as vl

    monkeypatch.setattr(vl, "vllm_registry_archs", lambda: {"SomethingElse"})
    assert "vllm" not in select_backends(hw_a100, spec, registry())
