"""8-bit W8A8 checkpoints (int8 weights and int8 activations) as an opt-in precision on vLLM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from polyserve import quantized as Q
from polyserve.backends import get_backend
from polyserve.bench.ablation import strategy_variants
from polyserve.calibrate.workload import get_workload
from polyserve.cli import _quant_list
from polyserve.models import Config, ModelSpec
from polyserve.pipeline import allowed_quants
from tests.conftest import make_hw


def _config(weights_type: str = "int", activations: bool = True) -> dict:
    """A compressed-tensors quantization_config shaped like Red Hat's Qwen2.5 W8A8 uploads."""
    group = {"weights": {"num_bits": 8, "type": weights_type, "strategy": "channel", "dynamic": False},
             "targets": ["Linear"]}
    if activations:
        group["input_activations"] = {"num_bits": 8, "type": "int", "strategy": "token", "dynamic": True}
    return {"quantization_config": {"quant_method": "compressed-tensors", "format": "int-quantized",
                                    "config_groups": {"group_0": group}, "ignore": ["lm_head"]}}


class Hub:
    models = [("hugging-quants/Qwen2.5-3B-Instruct-w8a8", 50), ("RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8", 10_451),
              ("RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16", 1_039)]

    def list_models(self, search, sort=None, direction=None, limit=30):
        return [SimpleNamespace(id=i, downloads=d) for i, d in self.models]

    def model_info(self, repo_id, files_metadata=False):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=3_900_000_000)])


CONFIGS = {
    "hugging-quants/Qwen2.5-3B-Instruct-w8a8": _config(weights_type="float"),  # fp8 under a w8a8 name
    "RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8": _config(),
    "RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a16": _config(activations=False),
}


def test_the_finder_checks_the_config_not_the_name():
    found = Q.find_prequantized_repos(ModelSpec(hf_id="Qwen/Qwen2.5-3B-Instruct"), methods=["w8a8"], api=Hub(),
                                      fetch_config=CONFIGS.get)
    assert found["w8a8"].repo_id == "RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8"  # the preferred author's fp8 is not
    assert (found["w8a8"].bits, found["w8a8"].size_bytes) == (8, 3_900_000_000)
    assert Q.verified_bits("w8a8", _config(activations=False)["quantization_config"]) is None  # weight-only W8A16
    assert Q.verified_bits("w8a8", {"quant_method": "gptq", "bits": 8}) is None


def test_it_is_opt_in_like_4bit(hw_a100):
    supported = get_backend("vllm").precisions(hw_a100)
    assert "w8a8" in supported and "w8a8" not in get_backend("vllm").precisions(make_hw("gtx1080"))
    assert "w8a8" not in allowed_quants(supported, None)
    assert "w8a8" in allowed_quants(supported, ["auto", "w8a8"])
    assert _quant_list("auto,w8a8") == "auto,w8a8"


@pytest.mark.usefixtures("no_network")
def test_vllm_loads_it_from_its_repository_without_a_quantization_flag(hw_a100, spec, monkeypatch):
    monkeypatch.setattr(Q, "find_prequantized_repos", lambda s, methods=Q.PREQUANTIZED, **kw: {
        m: Q.PrequantizedRepo(repo_id=f"RedHatAI/{s.hf_id.split('/')[1]}-quantized.w8a8", method=m, bits=8,
                              size_bytes=3_900_000_000) for m in methods if m == "w8a8"})
    be = get_backend("vllm")
    pm = be.prepare(spec, hw_a100, quants=["bf16", "w8a8"])
    assert pm.weights_bytes["w8a8"] == 3_900_000_000 and pm.hf_paths["w8a8"].endswith("-quantized.w8a8")
    args = be.launch_spec(Config(backend="vllm", quant="w8a8", ctx=4096, batch=64), pm, 1).args
    assert "--quantization" not in args and "--dtype" not in args
    assert any(a.endswith("-quantized.w8a8") for a in args)


def test_the_ablation_weighs_it_against_the_unquantized_weights(hw_a100, prepared_vllm):
    pm = prepared_vllm.model_copy(update={"weights_bytes": {**prepared_vllm.weights_bytes, "w8a8": 3_900_000_000}})
    base = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9)

    def weights(pick):
        return [v.label for v in strategy_variants(pick, get_backend("vllm"), hw_a100, pm, get_workload("chat"))
                if v.strategy == "weights"]

    assert "+w8a8" in weights(base)
    (back,) = weights(base.model_copy(update={"quant": "w8a8"}))
    assert back.startswith("-w8a8 (")
