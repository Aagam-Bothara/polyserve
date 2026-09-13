"""load_pick must not depend on the order the filesystem lists a directory in."""

from __future__ import annotations

import json
from pathlib import Path

from polyserve.bench.ablation import load_pick
from polyserve.bench.compare import results_path
from polyserve.models import Config, ModelSpec


def _write(root: Path, hw_hash: str, quant: str) -> None:
    rows = [{"label": "polyserve", "config": Config(backend="vllm", quant=quant).model_dump()}]
    results_path(hw_hash, ModelSpec(hf_id="org/M"), "chat", "balanced", root).write_text(json.dumps({"rows": rows}))


def test_same_pick_whatever_order_the_directory_lists(tmp_path, monkeypatch):
    for h, q in (("aaaa", "bf16"), ("mmmm", "awq"), ("zzzz", "fp8")):
        _write(tmp_path, h, q)
    real_glob = Path.glob
    picks = []
    for order in (lambda xs: xs, lambda xs: xs[::-1]):  # the listing order a filesystem may return
        monkeypatch.setattr(Path, "glob", lambda self, pat, _o=order: iter(_o(sorted(real_glob(self, pat)))))
        picks.append((load_pick(tmp_path, "org/M", "chat", "balanced").quant,
                      load_pick(tmp_path, "org/M", "chat", "balanced", hw_hash="mmmm").quant))
    assert picks[0] == picks[1] == ("bf16", "awq")  # by name when no machine matches; this machine first
