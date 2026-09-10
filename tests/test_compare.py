from __future__ import annotations

import json

import pytest

from polyserve.bench import ComparisonResult, compare, ollama_tag_for, reference_configs, to_markdown
from polyserve.bench.compare import save
from polyserve.bench.references import OllamaReference
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config, Profile, TrialMetrics, TrialResult
from polyserve.pipeline import prepare_and_plan, select
from tests.test_objectives_and_search import FakeRunner


def test_ollama_tag_heuristic():
    assert ollama_tag_for("meta-llama/Llama-3.2-3B-Instruct") == "llama3.2:3b"
    assert ollama_tag_for("meta-llama/Llama-3.1-8B-Instruct") == "llama3.1:8b"
    assert ollama_tag_for("Qwen/Qwen2.5-3B-Instruct") == "qwen2.5:3b"
    assert ollama_tag_for("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2.5:1.5b"
    assert ollama_tag_for("someone/CustomModel") is None


@pytest.mark.usefixtures("no_network")
def test_reference_configs_are_stock_defaults(hw_a100, spec):
    candidates, reg = select(hw_a100, spec)
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    refs = reference_configs(hw_a100, planned.prepared, reg)
    assert set(refs) == {"vllm-default", "sglang-default", "llamacpp-cuda-default"}
    v = refs["vllm-default"]
    assert v.ctx == 131072 and v.batch == 256 and v.gpu_memory_utilization == 0.90 and v.quant == "bf16"
    lc = refs["llamacpp-cuda-default"]
    assert lc.quant == "Q4_K_M" and lc.ctx == 4096 and lc.batch == 1 and lc.n_batch == 2048 and lc.n_gpu_layers == 29


def test_ollama_reference_launch_and_hooks(hw_a100, prepared_vllm, monkeypatch):
    import polyserve.bench.references as refs

    monkeypatch.setattr(refs, "ollama_binary", lambda: "/usr/local/bin/ollama")
    o = OllamaReference("qwen2.5:3b")
    assert o.available(hw_a100)
    ls = o.launch_spec(o.default_config(hw_a100, prepared_vllm), prepared_vllm, 11500)
    assert ls.args == ["/usr/local/bin/ollama", "serve"] and ls.env["OLLAMA_HOST"] == "127.0.0.1:11500"
    hooks = o.workload_hooks(hw_a100, prepared_vllm)
    assert hooks.model_name == "qwen2.5:3b" and hooks.health_path == "/" and hooks.tokenizer_id == prepared_vllm.spec.hf_id
    monkeypatch.setattr(refs, "ollama_binary", lambda: None)
    assert not o.available(hw_a100)


class CompareRunner(FakeRunner):
    """FakeRunner plus per-level metrics so SLO scoring has something to choose between."""

    def run(self, cfg: Config, stage: str) -> TrialResult:
        res = super().run(cfg, stage)
        if not res.ok:
            return res
        m = res.metrics
        levels = {}
        for c in (1, 4, 8):
            levels[str(c)] = TrialMetrics(tok_s=m.tok_s * c / 8, ttft_ms=m.ttft_ms * c / 4, tpot_ms=m.tpot_ms,
                                          joules_per_token=m.joules_per_token, requests=16, output_tokens=2048,
                                          concurrency=c)
        m.by_concurrency = levels
        return res


@pytest.mark.usefixtures("no_network")
def test_compare_measures_polyserve_and_references(tmp_path, hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    from polyserve.pipeline import calibrate

    profile: Profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    assert profile.calibration_trials > 0 and profile.calibration_seconds >= 0
    seen = []
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                     runner=CompareRunner(), progress=lambda label, row: seen.append((label, row is not None)))
    labels = [r.label for r in result.rows]
    assert labels == ["polyserve", "vllm-default"]
    ps = result.polyserve_row
    assert ps.ok and ps.scored_concurrency in (1, 4, 8) and ps.meets_slo is not None
    assert ps.calibration_trials == profile.calibration_trials
    assert ("polyserve", False) in seen and ("vllm-default", True) in seen
    # Stock vLLM asks for ctx 131072 x batch 256: in the fake model that is a huge TTFT, so it misses the SLO.
    dflt = result.rows[1]
    assert dflt.ok and dflt.scored_ttft_ms is not None
    path = save(result, tmp_path)
    assert path.name.endswith("__default__balanced.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["rows"][0]["label"] == "polyserve" and data["workload_spec"]["prefill_tokens"] == 256
    md = to_markdown(ComparisonResult.model_validate(data))
    assert "**PolyServe**" in md and "vllm-default" in md and "| SLO |" in md


@pytest.mark.usefixtures("no_network")
def test_compare_records_failed_reference(tmp_path, hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    from polyserve.pipeline import calibrate

    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())

    class OomDefaults(CompareRunner):
        def run(self, cfg, stage):
            if stage.endswith("vllm-default"):
                return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error="CUDA OOM")
            return super().run(cfg, stage)

    result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=OomDefaults())
    d = result.rows[1]
    assert not d.ok and d.error == "CUDA OOM" and d.scored_tok_s is None
    assert "failed: CUDA OOM" in to_markdown(result)
