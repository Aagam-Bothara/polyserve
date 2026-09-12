"""Prefill/decode optimisation: phase knobs in one engine (default) and disaggregated engines.

No GPU needed. Disaggregated serving is exercised end to end with fake engine processes and fake
KV-transfer upstreams that speak vLLM's kv_transfer_params handshake.
"""

from __future__ import annotations

import json
from typing import List

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from polyserve import disagg as D
from polyserve.backends import get_backend
from polyserve.calibrate.objectives import Constraints, pick
from polyserve.calibrate.search import StagedSearch
from polyserve.calibrate.workload import Workload, get_workload
from polyserve.models import Config, DisaggSpec, TrialMetrics, TrialResult
from polyserve.power import PowerSetting
from tests.conftest import make_hw
from tests.test_measure_and_proxy import _Server
from tests.test_objectives_and_search import FakeRunner, _feasible_a100
from tests.test_power import FakeController
from tests.test_supervisor import FakeBackend

KV_BACK = {"do_remote_prefill": True, "do_remote_decode": False, "remote_engine_id": "eng-1",
           "remote_block_ids": [1, 2, 3], "remote_host": "127.0.0.1", "remote_port": 5600}


def two_gpus(kind: str = "a100"):
    hw = make_hw(kind)
    hw.gpus.append(hw.gpus[0].model_copy(update={"index": 1, "uuid": "GPU-second"}))
    return hw


# =========================================================================== option 2: one engine


def test_prefill_knob_reaches_every_engine(hw_a100, prepared_vllm, prepared_llamacpp, monkeypatch):
    import polyserve.backends.llamacpp as lc

    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/opt/llama-server")
    v = get_backend("vllm").launch_spec(Config(backend="vllm", quant="fp8", ctx=4096, batch=64, prefill_budget=16384),
                                        prepared_vllm, 9000).args
    assert v[v.index("--max-num-batched-tokens") + 1] == "16384"
    s = get_backend("sglang").launch_spec(Config(backend="sglang", quant="bf16", ctx=4096, batch=64, prefill_budget=2048),
                                          prepared_vllm, 9001).args
    assert s[s.index("--chunked-prefill-size") + 1] == "2048"
    c = get_backend("vllm-cpu").launch_spec(Config(backend="vllm-cpu", quant="bf16", ctx=4096, batch=16,
                                                   prefill_budget=2048), prepared_vllm, 9002).args
    assert c[c.index("--max-num-batched-tokens") + 1] == "2048"
    ll = get_backend("llamacpp-cuda").launch_spec(
        Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=29, n_batch=512,
               prefill_budget=2048), prepared_llamacpp, 9003).args
    assert ll[ll.index("-ub") + 1] == "2048" and ll[ll.index("-b") + 1] == "2048"  # -b raised to cover -ub
    plain = get_backend("vllm").launch_spec(Config(backend="vllm", quant="fp8", ctx=4096, batch=64), prepared_vllm, 1)
    assert "--max-num-batched-tokens" not in plain.args  # unset = the engine's own default


def test_prefill_variants_respect_engine_rules():
    v = get_backend("vllm").prefill_variants(Config(backend="vllm", quant="fp8", ctx=4096, batch=256))
    assert [c.prefill_budget for c in v] == [2048, 8192, 16384]
    big = get_backend("vllm").prefill_variants(Config(backend="vllm", quant="fp8", ctx=4096, batch=4096))
    assert [c.prefill_budget for c in big] == [8192, 16384]  # vLLM needs budget >= max_num_seqs
    ll = get_backend("llamacpp-cuda").prefill_variants(
        Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_batch=512))
    assert [(c.prefill_budget, c.n_batch) for c in ll] == [(256, 512), (1024, 1024), (2048, 2048)]
    assert all(c.key() != ll[0].key() for c in ll[1:])
    assert Config(backend="vllm", quant="fp8", prefill_budget=8192).key().endswith("/pb8192")


def test_tpot_ceiling_protects_the_decode_phase():
    fast = TrialResult(config=Config(backend="vllm", quant="fp8", batch=256), stage="t",
                       metrics=TrialMetrics(tok_s=1200, ttft_ms=40, tpot_ms=80, requests=48, output_tokens=6000))
    steady = TrialResult(config=Config(backend="vllm", quant="fp8", batch=64), stage="t",
                         metrics=TrialMetrics(tok_s=1000, ttft_ms=40, tpot_ms=20, requests=48, output_tokens=6000))
    assert pick([fast, steady], "balanced", Constraints())[0] is fast
    w, notes = pick([fast, steady], "balanced", Constraints(tpot_ceiling_ms=50))
    assert w is steady  # the faster config streams tokens too slowly for each user
    w, notes = pick([fast], "throughput", Constraints(tpot_ceiling_ms=50))
    assert w is fast and any("least-violating" in n for n in notes)
    unmeasured = steady.model_copy(update={"metrics": steady.metrics.model_copy(update={"tpot_ms": None})})
    assert pick([unmeasured], "balanced", Constraints(tpot_ceiling_ms=50))[1][0].startswith("winner")


def test_workload_presets_carry_a_decode_slo():
    assert get_workload("chat").tpot_ceiling_ms == 50 and get_workload("generation").tpot_ceiling_ms == 50
    assert get_workload("default").tpot_ceiling_ms == 100 and get_workload("high-concurrency").tpot_ceiling_ms == 150
    assert get_workload("chat").spec()["tpot_ceiling_ms"] == 50
    assert "TPOT ceiling 50 ms" in get_workload("chat").describe()
    assert get_workload("default").spec() == Workload().spec()


class PrefillFakeRunner(FakeRunner):
    """A mid-sized prefill budget is 10% faster; the smallest starves prefill, the largest stalls decode."""

    def run(self, cfg: Config, stage: str) -> TrialResult:
        res = super().run(cfg.model_copy(update={"prefill_budget": None}), stage)
        if not res.ok:
            return res
        factor = {None: 1.0, 2048: 0.95, 8192: 1.10, 16384: 1.0}[cfg.prefill_budget]
        m = res.metrics.model_copy(update={"tok_s": res.metrics.tok_s * factor})
        return TrialResult(config=cfg, stage=stage, metrics=m)


def test_staged_search_tunes_the_prefill_knob_on_the_leader(hw_a100, prepared_vllm):
    feasible = _feasible_a100(hw_a100, prepared_vllm)
    be = get_backend("vllm")
    search = StagedSearch(objective="throughput", runner=PrefillFakeRunner(), prefill_variants=be.prefill_variants)
    winner, _ = search.run(feasible)
    rows = [r for r in search.results if r.stage == "prefill"]
    assert len(rows) == 3 and {r.config.prefill_budget for r in rows} == {2048, 8192, 16384}
    assert len({r.config.base_key().replace("/pb" + str(r.config.prefill_budget), "") for r in rows}) == 1
    assert winner.config.prefill_budget == 8192 and winner.config.quant == "fp8"


@pytest.mark.usefixtures("no_network")
def test_calibration_tunes_prefill_by_default_and_can_skip_it(hw_a100, spec):
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    on = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    assert any(r.stage == "prefill" for r in on.calibration_table)
    off = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner(), phase_tuning=False)
    assert not any(r.stage == "prefill" for r in off.calibration_table)


def test_predictor_respects_a_tpot_ceiling(hw_rtx4090, prepared_vllm):
    from polyserve import predict as P

    pred = P.Predictor(hw_rtx4090, {"vllm": P.PerfParams(alpha=0.6, beta=0.35, overhead_s=0.004, fitted=True)})
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    one = pred.predict(prepared_vllm, cfg, 256, 128, 1)
    eight = pred.predict(prepared_vllm, cfg, 256, 128, 8)
    assert eight.tpot_ms > one.tpot_ms
    capped = pred.best_level(prepared_vllm, cfg, 256, 128, (1, 4, 8), tpot_ceiling_ms=(one.tpot_ms + eight.tpot_ms) / 2)
    assert capped.concurrency < 8 and capped.tpot_ms <= (one.tpot_ms + eight.tpot_ms) / 2
    assert pred.best_level(prepared_vllm, cfg, 256, 128, (1, 4, 8)).concurrency == 8


# =========================================================================== option 3: disaggregated


def test_check_support_explains_every_blocker(monkeypatch):
    monkeypatch.setattr(D, "connector_available", lambda pkg: False)
    assert "needs vLLM" in D.check_support(two_gpus(), "llamacpp-cuda")
    assert "two NVIDIA GPUs; 1 visible" in D.check_support(make_hw("a100"), "vllm")
    assert "0 visible" in D.check_support(make_hw("cpu"), "vllm")
    assert "pip install nixl" in D.check_support(two_gpus(), "vllm")
    assert "unknown KV connector" in D.check_support(two_gpus(), "vllm", "carrier-pigeon")
    assert D.check_support(two_gpus(), "vllm", "custom", override={"kv_connector": "LMCacheConnectorV1"}) is None
    monkeypatch.setattr(D, "connector_available", lambda pkg: True)
    assert D.check_support(two_gpus(), "vllm") is None


def test_candidate_pairs_shape_each_pool_for_its_phase(prepared_vllm):
    hw = two_gpus()
    base = Config(backend="vllm", quant="fp8", ctx=4096, batch=128, gpu_memory_utilization=0.9, power_limit_w=245)
    pairs = D.candidate_pairs(hw, get_backend("vllm"), prepared_vllm, base)
    assert len(pairs) == 4  # 2 prefill budgets x 2 decode batch sizes
    for p in pairs:
        assert p.prefill_gpu == 0 and p.decode_gpu == 1
        assert p.prefill.batch == 64 and p.prefill.prefill_budget in (8192, 16384)  # few seqs, big prompt chunks
        assert p.decode.prefill_budget == 2048 and p.decode.batch in (128, 256)  # many seqs, small prefill
        assert p.prefill.power_limit_w is None and p.decode.power_limit_w is None
        assert p.kv_transfer_config == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    tiny = two_gpus()
    tiny.gpus[1].vram_free_bytes = 2 * 1024**3  # the decode GPU cannot hold the model
    assert D.candidate_pairs(tiny, get_backend("vllm"), prepared_vllm, base) == []
    assert D.candidate_pairs(make_hw("a100"), get_backend("vllm"), prepared_vllm, base) == []


def test_vllm_engine_launch_for_each_phase(prepared_vllm):
    cfg = Config(backend="vllm", quant="fp8", ctx=4096, batch=64, prefill_budget=16384)
    spec = get_backend("vllm").disagg_launch_spec(cfg, prepared_vllm, 9100, "prefill",
                                                  {"kv_connector": "NixlConnector", "kv_role": "kv_both"}, 1, 5601)
    a = spec.args
    assert json.loads(a[a.index("--kv-transfer-config") + 1]) == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    assert a[a.index("--max-num-batched-tokens") + 1] == "16384" and a[a.index("--port") + 1] == "9100"
    assert spec.env == {"CUDA_VISIBLE_DEVICES": "1", "VLLM_NIXL_SIDE_CHANNEL_PORT": "5601"}
    with pytest.raises(NotImplementedError):
        get_backend("llamacpp-cuda").disagg_launch_spec(cfg, prepared_vllm, 1, "prefill", {}, 0, 0)


def test_request_bodies_follow_the_kv_transfer_handshake():
    body = {"model": "m", "prompt": "hi", "max_tokens": 64, "stream": True, "stream_options": {"include_usage": True}}
    pre = D.prefill_request_body(body)
    assert pre["max_tokens"] == 1 and pre["stream"] is False and "stream_options" not in pre
    assert pre["kv_transfer_params"]["do_remote_decode"] is True and pre["kv_transfer_params"]["do_remote_prefill"] is False
    assert body["max_tokens"] == 64 and "kv_transfer_params" not in body  # caller's request untouched
    dec = D.decode_request_body(body, {"kv_transfer_params": KV_BACK})
    assert dec["kv_transfer_params"] == KV_BACK and dec["max_tokens"] == 64 and dec["stream"] is True
    assert "kv_transfer_params" not in D.decode_request_body(body, {})
    chat = D.prefill_request_body({"messages": [], "max_completion_tokens": 50})
    assert chat["max_completion_tokens"] == 1


def _prefill_engine() -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    async def handler(req: Request):
        app.state.seen.append(await req.json())
        return {"id": "p", "choices": [{"index": 0, "text": "x"}], "kv_transfer_params": KV_BACK}

    app.add_api_route("/v1/completions", handler, methods=["POST"])
    app.add_api_route("/v1/chat/completions", handler, methods=["POST"])
    app.add_api_route("/health", lambda: {"ok": True}, methods=["GET"])
    return app


def _decode_engine() -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    async def handler(req: Request):
        body = await req.json()
        app.state.seen.append(body)
        n = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4)
        if body.get("stream"):
            async def gen():
                for _ in range(n):
                    yield f"data: {json.dumps({'choices': [{'text': 'y '}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"choices": [{"message": {"role": "assistant", "content": "y " * n}}], "usage": {"completion_tokens": n}}

    app.add_api_route("/v1/completions", handler, methods=["POST"])
    app.add_api_route("/v1/chat/completions", handler, methods=["POST"])
    app.add_api_route("/v1/models", lambda: {"data": [{"id": "decode-model"}]}, methods=["GET"])
    app.add_api_route("/health", lambda: {"ok": True}, methods=["GET"])
    return app


def test_router_splits_each_request_across_the_two_engines():
    pre_app, dec_app = _prefill_engine(), _decode_engine()
    with _Server(pre_app) as pre, _Server(dec_app) as dec:
        app = D.create_pd_app(pre.url, dec.url)
        with TestClient(app) as client:
            body = {"model": "m", "prompt": "hi", "max_tokens": 5, "stream": True,
                    "stream_options": {"include_usage": True}}
            with client.stream("POST", "/v1/completions", json=body, headers={"x-request-id": "r-1"}) as s:
                assert s.headers["content-type"].startswith("text/event-stream")
                lines = [ln for ln in s.iter_lines() if ln.startswith("data:")]
            assert len(lines) == 6 and lines[-1] == "data: [DONE]"
            p, d = pre_app.state.seen[-1], dec_app.state.seen[-1]
            assert p["max_tokens"] == 1 and p["stream"] is False and "stream_options" not in p
            assert p["kv_transfer_params"]["do_remote_decode"] is True
            assert d["kv_transfer_params"] == KV_BACK and d["max_tokens"] == 5 and d["stream_options"]

            r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "q"}],
                                                          "max_completion_tokens": 3})
            assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 3
            assert pre_app.state.seen[-1]["max_completion_tokens"] == 1
            assert client.get("/v1/models").json()["data"][0]["id"] == "decode-model"
            assert client.get("/health").status_code == 200
        down = D.create_pd_app("http://127.0.0.1:9", dec.url)
        with TestClient(down) as client:
            r = client.post("/v1/completions", json={"prompt": "hi"})
            assert r.status_code == 502 and "prefill engine unavailable" in r.json()["error"]["message"]


class FakePDBackend(FakeBackend):
    """The fake engine, launchable as either half of a pair; records what it was asked to run."""

    def __init__(self):
        super().__init__()
        self.pd_launches: List[str] = []

    def disagg_launch_spec(self, cfg, model, port, role, kv_transfer_config, gpu_index, side_channel_port):
        self.pd_launches.append(role)
        spec = self.launch_spec(cfg, model, port)
        spec.env.update({"PD_ROLE": role, "PD_GPU": str(gpu_index)})
        return spec


def _pd_spec(**decode_update) -> DisaggSpec:
    base = Config(backend="fake", quant="none", ctx=128, batch=1)
    return DisaggSpec(prefill=base.model_copy(update={"prefill_budget": 8192}),
                      decode=base.model_copy(update={"prefill_budget": 2048, **decode_update}),
                      kv_transfer_config={"kv_connector": "NixlConnector", "kv_role": "kv_both"})


def test_disagg_runner_measures_the_pair_through_the_router(tmp_path):
    be = FakePDBackend()
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    ctl = FakeController()
    runner = D.DisaggTrialRunner(be, None, make_hw("cpu"), wl, log_dir=tmp_path, startup_timeout=15,
                                 power_for_gpu=lambda i: ctl, power_settle_s=0)
    res = runner.run(_pd_spec(), "pd")
    assert res.ok, res.error
    assert res.disagg is not None and res.disagg.key() == _pd_spec().key() and res.metrics.output_tokens == 8
    assert be.pd_launches == ["prefill", "decode"]

    out = runner.sweep(_pd_spec(), [PowerSetting(), PowerSetting(power_limit_w=245)], "pd-power")
    assert [r.ok for r in out] == [True, True]
    assert out[1].config.power_limit_w == 245 and out[1].disagg.decode.power_limit_w == 245
    assert be.pd_launches == ["prefill", "decode"] * 2  # one pair for the whole sweep
    assert ctl.history == ["cap 245 W", "restore"] and ctl.applied is None


def test_disagg_supervisor_serves_and_caps_the_decode_gpu(tmp_path):
    be = FakePDBackend()
    ctl = FakeController()
    sup = D.DisaggSupervisor(be, _pd_spec(sm_clock_mhz=1185), None, log_dir=tmp_path, startup_timeout=15, power=ctl)
    sup.start()
    try:
        assert sup.healthy()
        st = sup.status()
        assert st["phases"] == "disaggregated" and st["alive"] and st["decode"]["gpu"] == 1
        assert st["power"] == {"applied": "clock <= 1185 MHz", "error": None}
        app = D.create_pd_app(sup.prefill_url, sup.decode_url, status_fn=sup.status)
        with TestClient(app) as client:
            r = client.post("/v1/completions", json={"model": "fake", "prompt": "hi", "max_tokens": 3})
            assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 3
    finally:
        sup.stop()
    assert ctl.applied is None and ctl.history[-1] == "restore"


class FakeDisaggRunner:
    def __init__(self, tok_s: float):
        self.tok_s = tok_s
        self.calls: List[str] = []

    def run(self, spec: DisaggSpec, stage: str) -> TrialResult:
        self.calls.append(spec.key())
        bonus = 1.05 if spec.decode.batch == 256 else 1.0
        m = TrialMetrics(tok_s=self.tok_s * bonus, ttft_ms=60, tpot_ms=6, requests=48, output_tokens=6000)
        return TrialResult(config=spec.decode, stage=stage, metrics=m, disagg=spec)


@pytest.fixture
def unified_base(no_network, spec, monkeypatch):
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    monkeypatch.setattr(D, "connector_available", lambda pkg: True)
    hw = two_gpus()
    candidates, reg = select(hw, spec, force="vllm")
    planned = prepare_and_plan(hw, spec, candidates, reg)
    base = calibrate(hw, spec, "throughput", planned, reg, runner=FakeRunner())
    return hw, reg, base


def test_auto_keeps_disaggregated_only_when_it_wins(unified_base, spec):
    hw, reg, base = unified_base
    ref = D._unified_reference(base)
    assert ref is not None
    faster = D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="auto",
                                       runner=FakeDisaggRunner(ref.metrics.tok_s * 2))
    assert faster.phases == "auto" and faster.disagg is not None and faster.disagg.decode.batch == 256
    assert faster.config == faster.disagg.decode and "--kv-transfer-config" in faster.launch_args
    assert any("disaggregated wins" in n for n in faster.notes)
    slow_runner = FakeDisaggRunner(ref.metrics.tok_s * 0.5)
    slower = D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="auto", runner=slow_runner)
    assert slower.disagg is None and slower.config == base.config
    assert any("auto: serving unified" in n for n in slower.notes)
    # Pairs are recorded either way. The base already runs 256 sequences, so there is one decode shape.
    assert len(slow_runner.calls) == 2
    assert len(slower.calibration_table) == len(base.calibration_table) + len(slow_runner.calls)


def test_disaggregated_is_served_when_asked_even_if_slower(unified_base, spec):
    hw, reg, base = unified_base
    ref = D._unified_reference(base)
    p = D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="disaggregated",
                                  runner=FakeDisaggRunner(ref.metrics.tok_s * 0.5))
    assert p.disagg is not None and any("unified wins" in n for n in p.notes)


@pytest.mark.usefixtures("no_network")
def test_unsupported_machines_fail_or_fall_back(spec, monkeypatch):
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    hw = make_hw("a100")  # one GPU
    candidates, reg = select(hw, spec, force="vllm")
    base = calibrate(hw, spec, "throughput", prepare_and_plan(hw, spec, candidates, reg), reg, runner=FakeRunner())
    with pytest.raises(RuntimeError, match="two NVIDIA GPUs"):
        D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="disaggregated", runner=FakeDisaggRunner(1))
    auto = D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="auto", runner=FakeDisaggRunner(1))
    assert auto.disagg is None and any("two NVIDIA GPUs" in n and "serving unified" in n for n in auto.notes)


def test_profiles_are_cached_per_phase_mode(tmp_home, hw_a100, spec, prepared_vllm):
    from polyserve import cache
    from tests.test_cache_and_pipeline import _profile

    p = _profile(hw_a100, spec, prepared_vllm)
    cache.save(p)
    path = cache.save(p.model_copy(update={"phases": "auto"}))
    assert path.name == "balanced-phases-auto.json"
    assert cache.load(hw_a100, spec, "balanced", phases="auto").phases == "auto"
    assert cache.load(hw_a100, spec, "balanced").phases == "unified"
    assert cache.load(hw_a100, spec, "balanced", phases="disaggregated") is None


def test_cli_phase_options():
    from polyserve.cli import app

    r = CliRunner().invoke(app, ["bench", "x/y", "--phases", "sideways"])
    assert r.exit_code != 0
    r = CliRunner().invoke(app, ["workloads"])
    assert r.exit_code == 0 and "TPOT" in r.output


def test_report_draws_polyserve_in_its_own_colour():
    from polyserve.bench.report import RUNTIME_COLOR, render_svg
    from tests.test_report import _result, _row

    svg = render_svg([_result("A100", "m/a", "chat", [_row("polyserve", "vllm", 1500, 50),
                                                      _row("vllm-default", "vllm", 1000, 60)])])
    assert f'fill="{RUNTIME_COLOR["polyserve"]}" stroke' in svg  # filled PolyServe point, not vLLM blue
