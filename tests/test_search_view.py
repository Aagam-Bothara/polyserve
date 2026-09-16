from __future__ import annotations

import io
from typing import Optional

from rich.console import Console
from typer.testing import CliRunner

from polyserve import __version__
from polyserve.cli import app
from polyserve.models import Config, Profile, TrialMetrics, TrialResult
from polyserve.searchview import SearchView, stage_counts


def _cfg(batch: int = 64, spec: Optional[str] = None) -> Config:
    return Config(backend="vllm", quant="bf16", ctx=8192, batch=batch, gpu_memory_utilization=0.95, spec_decode=spec)


def _trial(stage: str, tok_s: float, batch: int = 64, spec: Optional[str] = None, ok: bool = True) -> TrialResult:
    m = TrialMetrics(tok_s=tok_s, ttft_ms=80.0, ttft_p95_ms=95.0, tpot_ms=6.0, requests=32, output_tokens=2048,
                     concurrency=8)
    # A trial carries its per-level metrics; the objective scores the level it would serve at.
    metrics = m.model_copy(update={"by_concurrency": {"8": m}}) if ok else TrialMetrics()
    return TrialResult(config=_cfg(batch, spec), stage=stage, metrics=metrics, launched=ok,
                       error=None if ok else "CUDA out of memory")


def _console() -> Console:
    """A console that is not a terminal: what a redirected run looks like."""
    return Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)


def test_trace_draws_one_row_per_stage_and_names_the_leader():
    trials = [_trial("quant", 700), _trial("batch", 705, batch=16), _trial("spec", 1138, spec="draft:x:4")]
    console = _console()
    view = SearchView.from_trials(trials, console, winner=trials[-1].config.key())
    console.print(view.render())
    out = console.file.getvalue()

    assert stage_counts(view) == {"quant": 1, "batch": 1, "spec": 1}
    for stage in ("quant", "batch", "spec"):
        assert stage in out
    assert "1138" in out
    assert view.leader is not None and view.leader.key == trials[-1].config.key()


def test_a_failed_trial_is_drawn_but_never_leads():
    view = SearchView.from_trials([_trial("quant", 700), _trial("quant", 0, batch=256, ok=False)], _console())

    assert stage_counts(view) == {"quant": 2}
    assert view.leader is not None and view.leader.tok_s == 700


def test_redirected_output_keeps_the_per_trial_lines():
    """The pods and CI grep the log lines, so a non-terminal run must not switch to the picture."""
    seen = []
    console = _console()
    view = SearchView(console, fallback=lambda stage, cfg, res: seen.append((stage, res is None)))
    t = _trial("spec", 900)

    view.progress("spec", t.config, None)
    view.progress("spec", t.config, t)

    assert seen == [("spec", True), ("spec", False)]
    assert view.stages == [] and console.file.getvalue() == ""


def test_a_running_trial_becomes_the_same_trial_when_it_lands():
    view = SearchView(_console(), graphical=True)
    view.live_enabled = False
    t = _trial("prefill", 1149)

    view.progress("prefill", t.config, None)
    assert stage_counts(view) == {"prefill": 1} and view.leader is None

    view.progress("prefill", t.config, t)
    assert stage_counts(view) == {"prefill": 1}
    assert view.leader is not None and view.leader.tok_s == 1149


def test_the_live_picture_draws_on_a_terminal():
    """The path a user actually sees: the live frame starts on the first trial and the last one stays put."""
    console = Console(file=io.StringIO(), width=100, force_terminal=True, color_system=None)
    view = SearchView(console, graphical=True)
    t = _trial("spec", 924, spec="suffix:24")

    view.progress("spec", t.config, None)
    view.progress("spec", t.config, t)
    view.on_stage("tuning vllm as well")
    view.stop()

    out = console.file.getvalue()
    assert "spec" in out and "924" in out
    assert "tuning vllm as well" in out and view.live is None


def test_profiles_trace_redraws_a_saved_search(tmp_path, hw_a100):
    profile = Profile(polyserve_version=__version__, hardware_hash="test-hash", hardware=hw_a100, model_id="org/model",
                      objective="balanced", backend="vllm", backend_version="1.0", config=_cfg(),
                      launch_args=["vllm", "serve", "org/model"],
                      calibration_table=[_trial("quant", 700), _trial("spec", 1138, spec="draft:x:4")],
                      notes=["sglang was not tuned further"])
    path = tmp_path / "profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")

    r = CliRunner().invoke(app, ["profiles", "--trace", str(path)])

    assert r.exit_code == 0, r.output
    assert "spec" in r.output and "1138" in r.output


def test_profiles_trace_rejects_an_unknown_name(tmp_home):
    r = CliRunner().invoke(app, ["profiles", "--trace", "no-such-model"])

    assert r.exit_code == 2
