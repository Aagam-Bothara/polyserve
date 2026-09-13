"""Calibration time on real text: fewer requests per level, and a backend far behind is not measured further."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from polyserve.calibrate.search import StagedSearch
from polyserve.calibrate.workload import get_workload
from polyserve.models import Config, TrialMetrics, TrialResult

VLLM = Config(backend="vllm", quant="bf16", ctx=4096, batch=64)
GGUF = [Config(backend="llamacpp-cuda", quant=q, ctx=4096, batch=4, n_gpu_layers=37, n_batch=512)
        for q in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")]


class QuantRunner:
    def __init__(self, tok_by_quant: Dict[str, float]):
        self.tok = tok_by_quant

    def run(self, cfg: Config, stage: str) -> TrialResult:
        m = TrialMetrics(tok_s=self.tok[cfg.quant], ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048,
                         concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


def _quant_trials(tok: Dict[str, float]) -> StagedSearch:
    search = StagedSearch(objective="balanced", runner=QuantRunner(tok))
    search.run([VLLM, *GGUF])
    return search


def test_a_backend_far_behind_the_leader_is_not_measured_further():
    search = _quant_trials({"bf16": 1677.4, "Q4_K_M": 470.5, "Q5_K_M": 467.9, "Q6_K": 423.8, "Q8_0": 396.4})
    assert [r.config.quant for r in search.results if r.stage == "quant"] == ["bf16", "Q4_K_M"]  # A40, sharegpt
    assert any("skipped llamacpp-cuda/Q5_K_M" in n and "under 50% of vllm's 1677" in n for n in search.notes)
    fresh = StagedSearch(objective="balanced", runner=QuantRunner(
        {"bf16": 1677.4, "Q4_K_M": 470.5, "Q5_K_M": 467.9, "Q6_K": 423.8, "Q8_0": 396.4}))
    assert fresh.stage_quant([VLLM, *GGUF]) == [("vllm", "bf16")]  # and it is not carried into later stages


def test_a_competitive_backend_keeps_all_its_quants():
    search = _quant_trials({"bf16": 467.1, "Q4_K_M": 418.8, "Q5_K_M": 443.5, "Q6_K": 371.2, "Q8_0": 372.2})
    assert len([r for r in search.results if r.stage == "quant"]) == 5  # A40, code-edit: within 25%


def test_real_text_levels_send_a_few_rounds_per_slot():
    sharegpt = replace(get_workload("sharegpt"), prompts=["p"] * 64)
    assert [sharegpt.level_requests(c) for c in sharegpt.concurrencies] == [8, 16, 64]
    chat = get_workload("chat")
    assert chat.level_requests(1) == len(chat.prompts)  # synthetic presets unchanged
    assert "requests_per_slot" not in chat.spec() and sharegpt.spec()["requests_per_slot"] == 2


def test_run_trial_sends_the_capped_number_at_each_level(monkeypatch):
    import polyserve.calibrate.measure as ms
    from polyserve.calibrate.tokens import TokenCounter
    from polyserve.calibrate.workload import Workload
    from polyserve.backends.base import LlmtraceHooks

    sent = []

    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        sent.append((concurrency, len(workload.prompts)))
        return [ms.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.05, tokens=4)] * len(workload.prompts)

    monkeypatch.setattr(ms, "_drive", fake_drive)
    wl = Workload(n_prompts=32, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4, 16), requests_per_slot=2)
    m = ms.run_trial("http://unused", LlmtraceHooks(model_name="m", gpu_ids=[]), wl, warmup=False,
                     counter=TokenCounter())
    assert sent == [(1, 8), (4, 8), (16, 32)] and m.requests == 48
