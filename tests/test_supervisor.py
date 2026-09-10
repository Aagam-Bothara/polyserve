"""Launch a real subprocess backend through Process / Supervisor / SubprocessTrialRunner."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import httpx
import pytest

from polyserve.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, free_port
from polyserve.calibrate.search import SubprocessTrialRunner
from polyserve.calibrate.workload import Workload
from polyserve.memory import MemoryModel
from polyserve.models import Config
from polyserve.serve.supervisor import Supervisor

FAKE = Path(__file__).parent / "fake_server.py"


class FakeBackend(BaseBackend):
    name = "fake"

    def __init__(self, extra_args: List[str] | None = None):
        self.extra_args = extra_args or []

    def memory_model(self, hw):
        return MemoryModel(runtime_workspace=0, kv_tokens_fn=lambda c: 0, device="cpu")

    def launch_spec(self, cfg: Config, model, port: int) -> LaunchSpec:
        return LaunchSpec(args=[sys.executable, str(FAKE), "--port", str(port), *self.extra_args, *cfg.extra.get("args", [])])

    def workload_hooks(self, hw, model) -> LlmtraceHooks:
        return LlmtraceHooks(model_name="fake", gpu_ids=[], process_memory=True)


def _cfg(**extra) -> Config:
    return Config(backend="fake", quant="none", ctx=128, batch=1, extra=extra)


def test_process_starts_becomes_ready_and_stops(tmp_path):
    be = FakeBackend(["--startup-delay", "0.3"])
    port = free_port()
    proc = be.launch(_cfg(), None, port, log_path=tmp_path / "p.log")
    try:
        assert proc.alive()
        assert proc.wait_ready(timeout=15, poll=0.1)
        assert httpx.get(f"http://127.0.0.1:{port}/v1/models").json()["data"][0]["id"] == "fake"
    finally:
        proc.stop()
    assert not proc.alive()
    assert "listening" in proc.tail_log()


def test_process_reports_startup_failure(tmp_path):
    be = FakeBackend(["--fail"])
    proc = be.launch(_cfg(), None, free_port(), log_path=tmp_path / "f.log")
    try:
        assert not proc.wait_ready(timeout=10, poll=0.1)
        assert proc.returncode() == 3
        assert "refusing" in proc.tail_log()
    finally:
        proc.stop()


def test_supervisor_restarts_after_crash(tmp_path):
    be = FakeBackend(["--die-after", "1.0"])
    sup = Supervisor(be, _cfg(), None, log_path=tmp_path / "s.log", startup_timeout=15, health_interval=0.3,
                     max_restarts=2)
    sup.start()
    try:
        first_pid = sup.process.pid
        assert sup.healthy()
        deadline = time.time() + 20
        while time.time() < deadline and sup.restarts == 0:
            time.sleep(0.2)
        assert sup.restarts >= 1
        # After restart the new process should become healthy again (until it dies again).
        deadline = time.time() + 15
        while time.time() < deadline and not sup.healthy():
            time.sleep(0.1)
        assert sup.healthy()
        assert sup.process.pid != first_pid
        assert sup.status()["backend"] == "fake"
    finally:
        sup.stop()


def test_subprocess_trial_runner_end_to_end(tmp_path):
    be = FakeBackend()
    wl = Workload(n_prompts=3, prefill_tokens=16, decode_tokens=6, concurrencies=(1, 2))
    runner = SubprocessTrialRunner(backends={"fake": be}, models={"fake": None}, hw=None, workload=wl,
                                   log_dir=tmp_path, startup_timeout=15, request_timeout=30)
    res = runner.run(_cfg(), "quant")
    assert res.ok, res.error
    assert res.metrics.output_tokens == 3 * 6 * 2
    assert res.metrics.telemetry_source in ("psutil", "rapl", "none")
    assert list(tmp_path.glob("*.log"))

    bad = SubprocessTrialRunner(backends={"fake": FakeBackend(["--fail"])}, models={"fake": None}, hw=None,
                                workload=wl, log_dir=tmp_path, startup_timeout=10)
    res = bad.run(_cfg(), "quant")
    assert not res.ok and not res.launched and "failed to start" in res.error


@pytest.mark.skipif(sys.platform == "win32", reason="process-group kill is POSIX-only")
def test_process_group_is_used_on_posix():
    import os

    be = FakeBackend()
    proc = be.launch(_cfg(), None, free_port())
    try:
        assert os.getpgid(proc.pid) == proc.pid
    finally:
        proc.stop()
