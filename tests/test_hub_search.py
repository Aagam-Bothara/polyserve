"""Hub search across huggingface_hub versions: 1.x removed list_models(direction=...)."""

from __future__ import annotations

from types import SimpleNamespace

import polyserve.quantized as Q
from polyserve.gguf import list_hub_models
from polyserve.models import ModelSpec


class Hub1x:
    """list_models without `direction`, as in huggingface_hub 1.31 (found on a vLLM 0.29 install)."""

    def __init__(self, ids):
        self.ids = ids

    def list_models(self, search, sort=None, limit=30):
        return [SimpleNamespace(id=i, downloads=10) for i in self.ids]

    def model_info(self, repo_id, files_metadata=False):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=2_000_000_000)])


def test_search_works_with_and_without_direction():
    new = Hub1x(["Qwen/Qwen2.5-3B-Instruct-AWQ"])
    assert [m.id for m in list_hub_models(new, "q", 5)] == ["Qwen/Qwen2.5-3B-Instruct-AWQ"]

    class Hub0x(Hub1x):
        def list_models(self, search, sort=None, direction=None, limit=30):
            assert direction == -1  # older releases still get descending order explicitly
            return super().list_models(search, sort=sort, limit=limit)

    assert [m.id for m in list_hub_models(Hub0x(["a/b"]), "q", 5)] == ["a/b"]


def test_4bit_checkpoints_are_found_on_huggingface_hub_1x():
    found = Q.find_prequantized_repos(ModelSpec(hf_id="Qwen/Qwen2.5-3B-Instruct"), methods=["awq"],
                              api=Hub1x(["Qwen/Qwen2.5-3B-Instruct-AWQ"]),
                              fetch_config=lambda rid: {"quantization_config": {"quant_method": "awq", "bits": 4}})
    assert found["awq"].repo_id == "Qwen/Qwen2.5-3B-Instruct-AWQ" and found["awq"].size_bytes == 2_000_000_000
