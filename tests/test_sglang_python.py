"""SGLang in its own environment ($SGLANG_PYTHON): found, versioned and launched from there."""

from __future__ import annotations

import polyserve.hardware as H
from polyserve.backends import get_backend
from polyserve.models import Config


def test_sglang_in_its_own_environment_is_found_and_launched_from_there(monkeypatch, tmp_path, hw_a100,
                                                                          prepared_vllm):
    py = tmp_path / "sgl" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv("SGLANG_PYTHON", str(py))
    monkeypatch.setattr(H, "_sglang_version", lambda p: "0.5.2" if p == str(py) else None)
    sgl = H.probe_backends(hw_a100.gpus)["sglang"]
    assert sgl.available and sgl.version == "0.5.2"
    args = get_backend("sglang").launch_spec(Config(backend="sglang", quant="bf16"), prepared_vllm, 1).args
    assert args[:3] == [str(py), "-m", "sglang.launch_server"]


def test_an_sglang_python_without_sglang_or_missing_is_reported(monkeypatch, tmp_path, hw_a100):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setenv("SGLANG_PYTHON", str(py))
    monkeypatch.setattr(H, "_sglang_version", lambda p: None)  # that environment has no sglang
    sgl = H.probe_backends(hw_a100.gpus)["sglang"]
    assert not sgl.available and "SGLANG_PYTHON" in sgl.reason
    monkeypatch.setenv("SGLANG_PYTHON", str(tmp_path / "nope" / "python"))
    assert not H.probe_backends(hw_a100.gpus)["sglang"].available
