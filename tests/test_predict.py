from __future__ import annotations


from polyserve import predict as P
from polyserve.calibrate.search import StagedSearch
from polyserve.models import Config, Profile, TrialMetrics, TrialResult
from tests.conftest import make_hw
from tests.test_objectives_and_search import FakeRunner


def test_device_spec_lookup_and_fallbacks(hw_a100, hw_rtx4090, hw_gtx1080, hw_cpu):
    assert P.device_spec(hw_a100).tflops == 312.0 and P.device_spec(hw_a100).known
    assert P.device_spec(hw_rtx4090).mem_bw_gbs == 1008.0
    assert P.device_spec(hw_gtx1080).tflops == 8.9
    hw = make_hw("a100")
    hw.gpus[0].name = "NVIDIA Mystery 48GB"
    d = P.device_spec(hw)
    assert not d.known and 200 <= d.mem_bw_gbs <= 3000
    c = P.device_spec(hw_cpu)
    assert not c.known and c.mem_bw_gbs == 40.0 and c.tflops > 0


def _sit(prepared, cfg, lp=256, ld=128):
    return P.situation(prepared, cfg, lp, ld)


def test_prediction_is_physically_sensible(hw_rtx4090, prepared_vllm):
    dev = P.device_spec(hw_rtx4090)
    pr = P.prior("vllm")
    bf16 = Config(backend="vllm", quant="bf16", ctx=4096, batch=64)
    fp8 = Config(backend="vllm", quant="fp8", ctx=4096, batch=64)
    p1 = P.predict(_sit(prepared_vllm, bf16), bf16, dev, pr, concurrency=1)
    p8 = P.predict(_sit(prepared_vllm, bf16), bf16, dev, pr, concurrency=8)
    q8 = P.predict(_sit(prepared_vllm, fp8), fp8, dev, pr, concurrency=8)
    assert p8.tok_s > p1.tok_s and p8.in_flight == 8 and p1.memory_bound
    assert q8.tok_s > p8.tok_s  # half the weight bytes -> faster when memory bound
    assert p8.ttft_ms >= p1.ttft_ms  # no queueing with 64 slots: TTFT is the request's own prefill either way
    # Ballpark for a 3B model on a 4090: hundreds to a couple thousand tok/s at c=8.
    assert 200 < p8.tok_s < 5000 and 5 < p8.tpot_ms < 60
    # Queueing: 8 clients on 1 slot -> median request waits for 4 waves.
    one_slot = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=1)
    sit = P.Situation(params_count=3_200_000_000, kv_bytes_per_token=36_864, device_weights_bytes=2_000_000_000)
    q = P.predict(sit, one_slot, dev, P.prior("llamacpp-cuda"), concurrency=8)
    solo = P.predict(sit, one_slot, dev, P.prior("llamacpp-cuda"), concurrency=1)
    assert q.ttft_ms > 4 * solo.ttft_ms


def _synthetic_observations(dev, backend="vllm", truth=P.PerfParams(alpha=0.6, beta=0.35, overhead_s=0.004)):
    sit = P.Situation(params_count=3_200_000_000, kv_bytes_per_token=36_864, device_weights_bytes=6_400_000_000)
    obs = []
    for batch in (16, 64, 256):
        for ctx in (2048, 4096, 8192):
            cfg = Config(backend=backend, quant="bf16", ctx=ctx, batch=batch, gpu_memory_utilization=0.9)
            for c in (1, 4, 8):
                pr = P.predict(sit, cfg, dev, truth, c)
                obs.append(P.Observation(backend=backend, config=cfg, situation=sit, concurrency=c,
                                         tok_s=pr.tok_s * (1.03 if c == 4 else 0.98), ttft_ms=pr.ttft_ms * 1.02))
    return obs


def test_fit_recovers_synthetic_parameters(hw_rtx4090):
    dev = P.device_spec(hw_rtx4090)
    obs = _synthetic_observations(dev)
    p = P.fit(obs, dev, "vllm")
    assert p.fitted and p.n == len(obs)
    # alpha and overhead trade off in the memory-bound regime, so exact recovery is not identifiable;
    # what matters is that the fitted parameters reproduce the generating model's predictions.
    truth = P.PerfParams(alpha=0.6, beta=0.35, overhead_s=0.004)
    for o in obs:
        got = P.predict(o.situation, o.config, dev, p, o.concurrency)
        want = P.predict(o.situation, o.config, dev, truth, o.concurrency)
        assert abs(got.tok_s - want.tok_s) / want.tok_s < 0.10
        assert abs(got.ttft_ms - want.ttft_ms) / want.ttft_ms < 0.15
    ev = P.evaluate(obs, dev, "vllm")
    assert ev.mape_tok_s < 6 and ev.mape_ttft < 8
    assert ev.spearman_tok_s > 0.95
    assert ev.prior_mape_tok_s is not None
    assert P.fit([], dev, "vllm").fitted is False


def test_params_roundtrip_and_predictor(tmp_home, hw_rtx4090, prepared_vllm):
    from polyserve.hardware import hardware_hash

    hh = hardware_hash(hw_rtx4090)
    assert not P.Predictor(hw_rtx4090).is_fitted("vllm")
    fitted = P.PerfParams(alpha=0.6, beta=0.35, overhead_s=0.004, fitted=True, n=27, mape_tok_s=4.2)
    path = P.save_params(hh, {"vllm": fitted, "sglang": P.prior("sglang")})
    assert path == tmp_home / "perf-model.json"
    loaded = P.load_params(hh)
    assert set(loaded) == {"vllm"} and loaded["vllm"].fitted and loaded["vllm"].mape_tok_s == 4.2
    pred = P.Predictor(hw_rtx4090)
    assert pred.is_fitted("vllm") and not pred.is_fitted("sglang")
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    best = pred.best_level(prepared_vllm, cfg, 256, 128, (1, 4, 8), ttft_ceiling_ms=500)
    assert best.fitted and best.concurrency in (1, 4, 8) and best.ttft_ms <= 500


def test_observations_from_profile(hw_rtx4090, spec, prepared_vllm):
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    by = {str(c): TrialMetrics(tok_s=100 * c, ttft_ms=30 * c, concurrency=c, requests=16, output_tokens=100)
          for c in (1, 4, 8)}
    good = TrialResult(config=cfg, stage="quant", metrics=TrialMetrics(tok_s=800, ttft_ms=240, requests=48,
                                                                        output_tokens=300, by_concurrency=by))
    bad = TrialResult(config=cfg, stage="quant", metrics=TrialMetrics(), launched=False, error="oom")
    prof = Profile(polyserve_version="0", hardware_hash="h", hardware=hw_rtx4090, model_id=spec.hf_id,
                   objective="balanced", workload="chat", workload_spec={"prefill_tokens": 512, "decode_tokens": 128},
                   backend="vllm", backend_version="1", config=cfg, prepared=prepared_vllm, launch_args=[],
                   calibration_table=[good, bad])
    obs = P.observations_from_profile(prof)
    assert len(obs) == 3 and {o.concurrency for o in obs} == {1, 4, 8}
    assert obs[0].situation.prefill_tokens == 512 and obs[0].situation.device_weights_bytes == prepared_vllm.weights_bytes["bf16"]


def test_search_prunes_with_fitted_predictor(hw_a100, prepared_vllm):
    from polyserve.backends import get_backend
    from polyserve.memory import plan

    be = get_backend("vllm")
    feasible = [c for c, _ in plan(hw_a100, prepared_vllm, be.candidate_configs(hw_a100, prepared_vllm), be.memory_model(hw_a100))]
    # A predictor that thinks bf16 is hopeless (10x slower than fp8) and is "fitted".
    params = {"vllm": P.PerfParams(alpha=0.6, beta=0.35, overhead_s=0.004, fitted=True, n=30)}
    pred = P.Predictor(hw_a100, params)

    class Biased(P.Predictor):
        def best_level(self, model, cfg, prefill, decode, concurrencies, ttft_ceiling_ms=None):
            p = super().best_level(model, cfg, prefill, decode, concurrencies, ttft_ceiling_ms)
            if cfg.quant == "bf16":
                p.tok_s /= 10
            return p

    runner = FakeRunner()
    search = StagedSearch(objective="throughput", runner=runner, predictor=Biased(hw_a100, params),
                          models={"vllm": prepared_vllm}, prune_below=0.4)
    winner, notes = search.run(feasible)
    quants_tried = {r.config.quant for r in search.results if r.stage == "quant"}
    assert quants_tried == {"fp8"} and winner.config.quant == "fp8"
    assert any("skipped" in n and "bf16" in n for n in notes)
    # Without a fitted predictor nothing is pruned.
    search2 = StagedSearch(objective="throughput", runner=FakeRunner(), predictor=pred, models={"vllm": prepared_vllm})
    search2.predictor.params["vllm"].fitted = False
    search2.run(feasible)
    assert {r.config.quant for r in search2.results if r.stage == "quant"} == {"bf16", "fp8"}


def test_render_markdown(hw_rtx4090):
    dev = P.device_spec(hw_rtx4090)
    ev = P.evaluate(_synthetic_observations(dev), dev, "vllm")
    md = P.render_markdown({"vllm": ev}, dev)
    assert "| vllm | 27 |" in md and "Spearman" in md
    assert "priors" in P.render_markdown({}, dev)
