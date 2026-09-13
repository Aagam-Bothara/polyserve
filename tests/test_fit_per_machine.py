"""`polyserve fit` fits each machine's trials against that machine's GPU; start-up OOMs are recognised."""

from __future__ import annotations

from typer.testing import CliRunner

import polyserve.hardware as H
import polyserve.predict as P
from polyserve import cache
from polyserve.calibrate.search import mentions_oom
from polyserve.cli import app
from tests.conftest import make_hw
from tests.test_cache_and_pipeline import _profile


def test_fit_uses_each_machines_own_gpu(tmp_home, spec, prepared_vllm, monkeypatch):
    a100, ada = make_hw("a100"), make_hw("rtx4090")
    for hw in (a100, ada):
        cache.save(_profile(hw, spec, prepared_vllm))
    specs_used = []
    monkeypatch.setattr(H, "probe", lambda: a100)  # this machine
    monkeypatch.setattr(P, "observations_from_profile", lambda p: [p.hardware_hash])
    monkeypatch.setattr(P, "fit_all", lambda obs, dev: {})
    monkeypatch.setattr(P, "device_spec", lambda hw: specs_used.append(hw.gpu.name) or "dev")
    monkeypatch.setattr(P, "render_markdown", lambda evals, dev: "table")

    everywhere = CliRunner().invoke(app, ["fit", "--all"])
    assert everywhere.exit_code == 0, everywhere.output
    assert sorted(specs_used) == sorted([a100.gpu.name, ada.gpu.name])  # not this machine's GPU twice

    specs_used.clear()
    here = CliRunner().invoke(app, ["fit"])
    assert here.exit_code == 0 and specs_used == [a100.gpu.name]
    assert CliRunner().invoke(app, ["fit", "--all", "--apply"]).exit_code == 2


def test_a40_is_a_known_gpu_and_does_not_shadow_the_a4000():
    a40 = make_hw("a100")
    a40.gpus[0].name = "NVIDIA A40"
    spec = P.device_spec(a40)
    assert spec.known and spec.mem_bw_gbs == 696.0 and spec.tflops == 149.7
    a4000 = make_hw("a100")
    a4000.gpus[0].name = "NVIDIA RTX A4000"
    assert P.device_spec(a4000).mem_bw_gbs == 448.0


def test_a_fit_that_predicts_worse_than_the_defaults_is_not_kept():
    fitted = P.PerfParams(alpha=1.0, beta=0.27, overhead_s=0.0022, fitted=True)
    worse = P.choose(P.Evaluation(backend="vllm", n=219, mape_tok_s=28.5, mape_ttft=121.2, spearman_tok_s=0.88,
                                  params=fitted, prior_mape_tok_s=22.8))  # the A40 vLLM numbers
    assert worse.keeps_prior and worse.params == P.prior("vllm") and not worse.params.fitted
    better = P.choose(P.Evaluation(backend="llamacpp-cuda", n=117, mape_tok_s=22.9, mape_ttft=89.3,
                                   spearman_tok_s=0.89, params=fitted, prior_mape_tok_s=36.3))
    assert not better.keeps_prior and better.params is fitted
    assert "vllm (defaults kept)" in P.render_markdown({"vllm": worse}, P.DeviceSpec("A40", 696.0, 149.7, True))


def test_startup_oom_is_recognised():
    assert mentions_oom("... RuntimeError: CUDA out of memory occurred when warming up sampler with 512 dummy "
                        "requests ...")
    assert mentions_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 150.00 MiB")
    assert not mentions_oom("RuntimeError: Engine core initialization failed. See root cause above.")
