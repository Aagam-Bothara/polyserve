"""A reservation engine that runs out of memory at start-up is retried with less of the GPU reserved."""

from __future__ import annotations

from polyserve.calibrate.search import StagedSearch
from polyserve.models import Config, TrialMetrics, TrialResult

OOM = "out of memory at start-up; failed to start (rc=1)"


class StartupRunner:
    """Configs reserving at least `fails_at` of the GPU (every config with fail_all) fail to start with
    `error`; the rest run."""

    def __init__(self, fails_at: float = 0.95, error: str = OOM, fail_all: bool = False):
        self.fails_at, self.error, self.fail_all, self.ran = fails_at, error, fail_all, []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.ran.append(cfg)
        gmu = cfg.gpu_memory_utilization
        if self.fail_all or (gmu is not None and gmu >= self.fails_at - 1e-9):
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=self.error)
        m = TrialMetrics(tok_s=500.0, ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048, concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


VLLM_B512 = Config(backend="vllm", quant="bf16", ctx=8192, batch=512, gpu_memory_utilization=0.95, kv_dtype="fp8_e5m2")


def test_an_oom_at_start_up_is_retried_with_less_reserved():
    runner = StartupRunner()
    search = StagedSearch(objective="throughput", runner=runner)
    winner, notes = search.run([VLLM_B512])
    assert [c.gpu_memory_utilization for c in runner.ran] == [0.95, 0.90]
    assert winner.config.gpu_memory_utilization == 0.90
    assert any(not r.launched for r in search.results)  # the failed launch is kept for the memory report
    assert any("retried at gpu_memory_utilization 0.90" in n for n in notes)


def test_retries_stop_at_the_floor():
    runner = StartupRunner(fails_at=0.0)  # nothing ever starts
    StagedSearch(objective="throughput", runner=runner).run([VLLM_B512])
    assert [c.gpu_memory_utilization for c in runner.ran] == [0.95, 0.90, 0.85]


def test_other_start_up_failures_and_engines_without_a_reservation_are_not_retried():
    runner = StartupRunner(error="failed to start (rc=1)")  # not a memory failure
    StagedSearch(objective="throughput", runner=runner).run([VLLM_B512])
    assert len(runner.ran) == 1
    llama = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=8, n_gpu_layers=37, n_batch=512)
    runner = StartupRunner(fail_all=True)  # llama.cpp out of memory: nothing reserved to give back
    StagedSearch(objective="throughput", runner=runner).run([llama])
    assert len(runner.ran) == 1


def test_a_retry_the_memory_check_rejects_is_not_launched():
    runner = StartupRunner()
    StagedSearch(objective="throughput", runner=runner,
                 feasible_fn=lambda c: c.gpu_memory_utilization != 0.90).run([VLLM_B512])
    assert len(runner.ran) == 1
