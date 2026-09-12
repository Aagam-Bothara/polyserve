"""The five added optimization strategies: KV-cache quantization, 4-bit weights, prefix caching,
speculative decoding, multi-GPU layouts. No GPU or network needed."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import List

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from polyserve import quantized as Q
from polyserve import speculative as S
from polyserve.backends import get_backend, registry
from polyserve.calibrate.objectives import Constraints, pick
from polyserve.calibrate.search import StagedSearch
from polyserve.calibrate.workload import Workload, get_workload
from polyserve.memory import estimate, kv_cache_bytes
from polyserve.models import Config, ModelSpec, TrialMetrics, TrialResult
from polyserve.pipeline import PlanResult, SearchOptions, kv_variants_fn
from tests.conftest import make_hw
from tests.test_measure_and_proxy import _Server
from tests.test_objectives_and_search import FakeRunner, _feasible_a100
from tests.test_supervisor import FakeBackend


def two_gpus(kind: str = "a100"):
    hw = make_hw(kind)
    hw.gpus.append(hw.gpus[0].model_copy(update={"index": 1, "uuid": "GPU-second"}))
    return hw


def _words(text: str) -> int:
    return len(text.split())


# =========================================================================== 1. KV-cache quantization


def test_kv_bytes_follow_the_cache_type(prepared_llamacpp):
    base = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4)
    per_tok = prepared_llamacpp.arch.kv_bytes_per_token(1)  # K and V elements per token
    tokens = 4096 * 4
    assert kv_cache_bytes(prepared_llamacpp, base, tokens) == per_tok * 2 * tokens  # bf16
    assert kv_cache_bytes(prepared_llamacpp, base.model_copy(update={"kv_dtype": "q8_0"}), tokens) == int(
        per_tok * 34 / 32 * tokens)
    assert kv_cache_bytes(prepared_llamacpp, base.model_copy(update={"kv_dtype": "q4_0"}), tokens) == int(
        per_tok * 18 / 32 * tokens)
    assert kv_cache_bytes(prepared_llamacpp, base.model_copy(update={"kv_dtype": "fp8"}), tokens) == per_tok * tokens


def test_kv_flags_per_engine(prepared_vllm, prepared_llamacpp, monkeypatch):
    import polyserve.backends.llamacpp as lc

    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/opt/llama-server")
    v = get_backend("vllm").launch_spec(Config(backend="vllm", quant="bf16", kv_dtype="fp8"), prepared_vllm, 1).args
    assert v[v.index("--kv-cache-dtype") + 1] == "fp8"
    s = get_backend("sglang").launch_spec(Config(backend="sglang", quant="bf16", kv_dtype="fp8_e5m2"),
                                          prepared_vllm, 1).args
    assert s[s.index("--kv-cache-dtype") + 1] == "fp8_e5m2"
    cpu = get_backend("llamacpp-cpu")
    q = cpu.launch_spec(Config(backend="llamacpp-cpu", quant="Q4_K_M", batch=1, kv_dtype="q8_0"),
                        prepared_llamacpp, 1).args
    assert q[q.index("-ctk") + 1] == "q8_0" and q[q.index("-ctv") + 1] == "q8_0"
    assert q[q.index("-fa") + 1] == "on"  # quantized V needs flash attention, on CPU too
    plain = cpu.launch_spec(Config(backend="llamacpp-cpu", quant="Q4_K_M", batch=1), prepared_llamacpp, 1).args
    assert "-fa" not in plain


def test_which_engines_offer_a_quantized_cache(hw_a100, monkeypatch):
    import polyserve.backends.vllm as vl

    real = vl.importlib.util.find_spec
    monkeypatch.setattr(vl.importlib.util, "find_spec", lambda n: None if n == "flashinfer" else real(n))
    assert get_backend("vllm").kv_dtypes(hw_a100) == []  # Ampere without FlashInfer cannot run an fp8 cache
    monkeypatch.setattr(vl.importlib.util, "find_spec", lambda n: object() if n == "flashinfer" else real(n))
    assert get_backend("vllm").kv_dtypes(hw_a100) == ["fp8"]
    hopper = make_hw("a100")
    hopper.gpus[0].compute_capability = (9, 0)
    monkeypatch.setattr(vl.importlib.util, "find_spec", lambda n: None if n == "flashinfer" else real(n))
    assert get_backend("vllm").kv_dtypes(hopper) == ["fp8"]
    assert get_backend("sglang").kv_dtypes(hw_a100) == ["fp8_e5m2"]
    assert get_backend("llamacpp-cuda").kv_dtypes(hw_a100) == ["q8_0", "q4_0"]
    assert get_backend("vllm-cpu").kv_dtypes(make_hw("cpu-avx512")) == []


def test_kv_stage_adds_the_batch_a_smaller_cache_admits(hw_gtx1080, spec, prepared_llamacpp):
    plan = PlanResult(hw=hw_gtx1080, spec=spec, candidates=["llamacpp-cuda"],
                      prepared={"llamacpp-cuda": prepared_llamacpp})
    fn = kv_variants_fn(hw_gtx1080, registry(), plan)
    leader = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=8, n_gpu_layers=29, n_batch=512)
    keys = {(c.kv_dtype, c.batch) for c in fn(leader)}
    assert ("q8_0", 8) in keys and ("q4_0", 8) in keys
    assert ("q4_0", 16) in keys  # 16 slots fit on 8 GB only with a 4-bit cache
    assert ("q8_0", 16) not in keys  # still too big at 8 bits
    be = get_backend("llamacpp-cuda")
    mm = be.memory_model(hw_gtx1080)
    assert not estimate(hw_gtx1080, prepared_llamacpp, leader.model_copy(update={"batch": 16}), mm).feasible


def test_search_runs_extra_stages_on_the_leader(hw_a100, prepared_vllm):
    class KvFaster(FakeRunner):
        def run(self, cfg, stage):
            res = super().run(cfg.model_copy(update={"kv_dtype": "auto"}), stage)
            if res.ok and cfg.kv_dtype == "fp8":
                res = res.model_copy(update={"metrics": res.metrics.model_copy(update={"tok_s": res.metrics.tok_s * 1.2})})
            return res.model_copy(update={"config": cfg})

    search = StagedSearch(objective="throughput", runner=KvFaster(),
                          variant_stages=[("kv", lambda c: [c.model_copy(update={"kv_dtype": "fp8"})])])
    winner, _ = search.run(_feasible_a100(hw_a100, prepared_vllm))
    assert [r.stage for r in search.results].count("kv") == 1 and winner.config.kv_dtype == "fp8"


# =========================================================================== 2. 4-bit weights


class FakeHub:
    """list_models / model_info with just the fields the resolver reads."""

    def __init__(self):
        self.models = [
            ("Qwen/Qwen2.5-3B-Instruct-AWQ", 90_000), ("someone/Qwen2.5-3B-Instruct-AWQ-v2", 900_000),
            ("Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8", 50_000), ("Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4", 40_000),
            ("bartowski/Qwen2.5-3B-Instruct-GGUF", 2_000_000), ("Qwen/Qwen2.5-7B-Instruct-AWQ", 3_000_000),
        ]

    def list_models(self, search, sort=None, direction=None, limit=30):
        return [SimpleNamespace(id=i, downloads=d) for i, d in self.models]

    def model_info(self, repo_id, files_metadata=False):
        files = [SimpleNamespace(rfilename="model.safetensors", size=2_000_000_000),
                 SimpleNamespace(rfilename="README.md", size=5_000)]
        return SimpleNamespace(siblings=files)


CONFIGS = {
    "Qwen/Qwen2.5-3B-Instruct-AWQ": {"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}},
    "someone/Qwen2.5-3B-Instruct-AWQ-v2": {"quantization_config": {"quant_method": "awq", "bits": 4}},
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8": {"quantization_config": {"quant_method": "gptq", "bits": 8}},
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4": {"quantization_config": {"quant_method": "gptq", "bits": 4}},
}


def test_int4_resolver_prefers_the_author_and_verifies_bits():
    found = Q.find_int4_repos(ModelSpec(hf_id="Qwen/Qwen2.5-3B-Instruct"), api=FakeHub(), fetch_config=CONFIGS.get)
    assert found["awq"].repo_id == "Qwen/Qwen2.5-3B-Instruct-AWQ"  # the author beats a more downloaded fork
    assert found["awq"].group_size == 128 and found["awq"].size_bytes == 2_000_000_000
    assert found["gptq"].repo_id == "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"  # the 8-bit upload is rejected
    none = Q.find_int4_repos(ModelSpec(hf_id="org/Unrelated-Model"), api=FakeHub(), fetch_config=CONFIGS.get)
    assert none == {}


@pytest.mark.usefixtures("no_network")
def test_vllm_serves_a_4bit_checkpoint_under_a_stable_name(hw_a100, spec, monkeypatch):
    monkeypatch.setattr(Q, "find_int4_repos", lambda s, **kw: {
        "awq": Q.Int4Repo(repo_id="org/Llama-3.2-3B-Instruct-AWQ", method="awq", size_bytes=2_100_000_000)})
    be = get_backend("vllm")
    pm = be.prepare(spec, hw_a100)
    assert pm.weights_bytes["awq"] == 2_100_000_000 and pm.hf_paths == {"awq": "org/Llama-3.2-3B-Instruct-AWQ"}
    assert pm.weights_bytes["fp8"] * 2 == pm.weights_bytes["bf16"]
    args = be.launch_spec(Config(backend="vllm", quant="awq", ctx=4096, batch=64), pm, 1).args
    assert args[args.index("--model") + 1] == "org/Llama-3.2-3B-Instruct-AWQ"
    assert args[args.index("--served-model-name") + 1] == spec.hf_id
    assert "--quantization" not in args and "--dtype" not in args  # the checkpoint picks the kernel
    assert be.workload_hooks(hw_a100, pm).model_name == spec.hf_id


def test_precisions_track_the_gpu_generation(hw_a100):
    turing = make_hw("a100")
    turing.gpus[0].compute_capability = (7, 5)
    assert get_backend("vllm").precisions(turing) == ["fp16", "awq", "gptq"]
    assert get_backend("vllm").precisions(hw_a100) == ["bf16", "fp8", "awq", "gptq"]
    assert get_backend("vllm-cpu").precisions(make_hw("cpu-avx512")) == ["bf16"]


@pytest.mark.usefixtures("no_network")
def test_quant_option_restricts_what_calibration_may_choose(hw_a100, spec):
    from polyserve.pipeline import prepare_and_plan

    reg = registry()
    locked = prepare_and_plan(hw_a100, spec, ["vllm", "llamacpp-cuda"], reg, quants=["bf16"])
    assert set(locked.prepared["vllm"].weights_bytes) == {"bf16"}
    assert "none of --quant bf16" in locked.errors["llamacpp-cuda"]
    mixed = prepare_and_plan(hw_a100, spec, ["vllm", "llamacpp-cuda"], reg, quants=["fp8", "Q4_K_M"])
    assert set(mixed.prepared["vllm"].weights_bytes) == {"fp8"}
    assert set(mixed.prepared["llamacpp-cuda"].weights_bytes) == {"Q4_K_M"}
    assert {c.quant for c in mixed.all_feasible} == {"fp8", "Q4_K_M"}


# =========================================================================== 3. prefix caching


def test_shared_prefix_workload_builds_prompts_on_one_prefix():
    from polyserve.calibrate.tokens import TokenCounter

    wl = Workload(n_prompts=4, prefill_tokens=200, shared_prefix_tokens=150, decode_tokens=8)
    assert wl.prefix_text and all(p.startswith(wl.prefix_text) for p in wl.prompts)
    assert len({p[len(wl.prefix_text):] for p in wl.prompts}) == 4  # distinct after the prefix
    wl.fit_prompts(TokenCounter(encode=_words, name="words"))
    assert abs(_words(wl.prefix_text) - 150) <= 2
    assert all(abs(_words(p) - 200) <= 2 and p.startswith(wl.prefix_text) for p in wl.prompts)
    assert "(150 shared)" in wl.describe() and wl.spec()["shared_prefix_tokens"] == 150
    assert get_workload("chat-system").shared_prefix_tokens == 1536
    assert get_workload("rag-shared").shared_prefix_tokens == 5632
    assert get_workload("default").shared_prefix_tokens == 0


def test_warmup_leaves_the_real_prefix_cached(monkeypatch):
    import polyserve.calibrate.measure as M
    from polyserve.backends.base import LlmtraceHooks
    from polyserve.calibrate.tokens import TokenCounter

    seen: List[List[str]] = []

    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        seen.append(list(workload.prompts))
        return [M.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.05, tokens=4, token_source="usage")]

    monkeypatch.setattr(M, "_drive", fake_drive)
    wl = Workload(n_prompts=2, prefill_tokens=64, shared_prefix_tokens=48, decode_tokens=4, concurrencies=(1,))
    M.run_trial("http://x", LlmtraceHooks(), wl, warmup=True, counter=TokenCounter())
    warmup, real = seen[0], seen[1]
    assert all(p.startswith(wl.prefix_text) for p in warmup + real)


def test_prefix_cache_controls(prepared_vllm, prepared_llamacpp, monkeypatch):
    import polyserve.backends.llamacpp as lc

    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/opt/llama-server")
    v = get_backend("vllm")
    assert "--enable-prefix-caching" in v.launch_spec(Config(backend="vllm", quant="bf16", prefix_cache=True),
                                                      prepared_vllm, 1).args
    assert "--no-enable-prefix-caching" in v.launch_spec(Config(backend="vllm", quant="bf16", prefix_cache=False),
                                                         prepared_vllm, 1).args
    assert "--disable-radix-cache" in get_backend("sglang").launch_spec(
        Config(backend="sglang", quant="bf16", prefix_cache=False), prepared_vllm, 1).args
    ll = get_backend("llamacpp-cuda")
    base = Config(backend="llamacpp-cuda", quant="Q4_K_M", batch=4, n_gpu_layers=29)
    variants = ll.prefix_variants(base)
    assert [c.extra for c in variants] == [{"cache_reuse": 256}, {"kv_unified": True}]
    assert len({base.key(), *(c.key() for c in variants)}) == 3
    args = ll.launch_spec(variants[1], prepared_llamacpp, 1).args
    i = args.index("--kv-unified")
    assert i == len(args) - 1 or args[i + 1].startswith("-")  # a bare flag, no value
    reuse = ll.launch_spec(variants[0], prepared_llamacpp, 1).args
    assert reuse[reuse.index("--cache-reuse") + 1] == "256"


@pytest.mark.usefixtures("no_network")
def test_prefix_stage_only_for_shared_prefix_workloads(hw_gtx1080, spec, monkeypatch):
    import polyserve.backends.llamacpp as lc
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    def fake_materialize(self, m, quants):
        for q in quants:
            m.gguf_paths[q] = f"/models/{q}.gguf"
        return m

    monkeypatch.setattr(lc.LlamaCppBackend, "materialize", fake_materialize)
    candidates, reg = select(hw_gtx1080, spec, force="llamacpp-cuda")
    shared = get_workload("chat-system")
    plan = prepare_and_plan(hw_gtx1080, spec, candidates, reg, materialize=True, workload=shared)
    on = calibrate(hw_gtx1080, spec, "throughput", plan, reg, workload=shared, runner=FakeRunner())
    assert any(r.stage == "prefix" for r in on.calibration_table)
    off = calibrate(hw_gtx1080, spec, "throughput", plan, reg, workload=shared, runner=FakeRunner(),
                    options=SearchOptions(prefix_cache=False))
    assert not any(r.stage == "prefix" for r in off.calibration_table)
    assert all(r.config.prefix_cache is False for r in off.calibration_table)
    assert off.options == {"prefix": "off"}
    plain = prepare_and_plan(hw_gtx1080, spec, candidates, reg, materialize=True)
    default = calibrate(hw_gtx1080, spec, "throughput", plain, reg, runner=FakeRunner())
    assert not any(r.stage == "prefix" for r in default.calibration_table)


# =========================================================================== 4. speculative decoding


def test_draft_models_share_the_family_tokenizer():
    assert S.draft_for("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert S.draft_for("meta-llama/Llama-3.2-3B-Instruct") == "meta-llama/Llama-3.2-1B-Instruct"
    assert S.draft_for("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.2-1B-Instruct"
    assert S.draft_for("Qwen/Qwen2.5-0.5B-Instruct") is None  # nothing smaller to draft with
    assert S.draft_for("mistralai/Mistral-7B-Instruct-v0.3") is None
    assert S.parse("ngram:4") == ("ngram", None, 4)
    assert S.parse("draft:Qwen/Qwen2.5-0.5B-Instruct:16") == ("draft", "Qwen/Qwen2.5-0.5B-Instruct", 16)
    with pytest.raises(ValueError):
        S.parse("medusa:3")


def test_vllm_speculative_variants_and_flags(prepared_vllm):
    be = get_backend("vllm")
    base = Config(backend="vllm", quant="fp8", ctx=4096, batch=16)
    variants = be.spec_variants(base, prepared_vllm)
    assert [c.spec_decode for c in variants] == ["ngram:4", "draft:meta-llama/Llama-3.2-1B-Instruct:4"]
    args = be.launch_spec(variants[1], prepared_vllm, 1).args
    assert json.loads(args[args.index("--speculative-config") + 1]) == {
        "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 4}
    ngram = json.loads(be.launch_spec(variants[0], prepared_vllm, 1).args[-1])
    assert ngram["method"] == "ngram" and ngram["num_speculative_tokens"] == 4
    assert get_backend("vllm-cpu").spec_variants(base, prepared_vllm) == []


def test_llamacpp_speculation_needs_a_downloaded_draft(prepared_llamacpp, monkeypatch):
    import polyserve.backends.llamacpp as lc

    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/opt/llama-server")
    be = get_backend("llamacpp-cuda")
    base = Config(backend="llamacpp-cuda", quant="Q4_K_M", batch=4, n_gpu_layers=29)
    assert be.spec_variants(base, prepared_llamacpp) == []  # no draft resolved
    draft = "meta-llama/Llama-3.2-1B-Instruct"
    remote = prepared_llamacpp.model_copy(update={"draft_paths": {draft: "hf://org/repo/draft-Q8_0.gguf"}})
    assert be.spec_variants(base, remote) == []  # resolved but not downloaded
    local = prepared_llamacpp.model_copy(update={"draft_paths": {draft: "/models/draft-Q8_0.gguf"}})
    (v,) = be.spec_variants(base, local)
    args = be.launch_spec(v, local, 1).args
    assert args[args.index("-md") + 1] == "/models/draft-Q8_0.gguf" and args[args.index("--draft-max") + 1] == "16"
    assert "-ngld" in args
    with pytest.raises(RuntimeError, match="not materialized"):
        be.launch_spec(v, remote, 1)


def test_latency_objective_counts_the_whole_answer():
    fast_start = TrialResult(config=Config(backend="vllm", quant="fp8", batch=64), stage="t",
                             metrics=TrialMetrics(tok_s=900, ttft_ms=40, tpot_ms=20, requests=16, output_tokens=2048))
    speculative = TrialResult(config=Config(backend="vllm", quant="fp8", batch=64, spec_decode="ngram:4"), stage="t",
                              metrics=TrialMetrics(tok_s=800, ttft_ms=60, tpot_ms=5, requests=16, output_tokens=2048))
    w, _ = pick([fast_start, speculative], "latency", Constraints(tok_s_floor_abs=100))
    assert w is speculative  # 60 + 5 x 127 ms beats 40 + 20 x 127 ms


# =========================================================================== 5. multi-GPU layouts


def _counting_engine(name: str) -> FastAPI:
    app = FastAPI()
    app.state.count = 0

    async def completions(req: Request):
        body = await req.json()
        app.state.count += 1
        if body.get("stream"):
            async def gen():
                yield f"data: {json.dumps({'choices': [{'text': name}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"choices": [{"text": name}], "usage": {"completion_tokens": 1}}

    app.add_api_route("/v1/completions", completions, methods=["POST"])
    app.add_api_route("/health", lambda: {"ok": True}, methods=["GET"])
    return app


def test_load_balancer_spreads_requests_and_streams():
    from polyserve.layout import create_lb_app

    a, b = _counting_engine("a"), _counting_engine("b")
    with _Server(a) as sa, _Server(b) as sb:
        with TestClient(create_lb_app([sa.url, sb.url])) as client:
            for _ in range(10):
                assert client.post("/v1/completions", json={"prompt": "x"}).status_code == 200
            assert a.state.count == 5 and b.state.count == 5  # idle replicas alternate
            with client.stream("POST", "/v1/completions", json={"prompt": "x", "stream": True}) as s:
                assert s.headers["content-type"].startswith("text/event-stream")
                assert [ln for ln in s.iter_lines() if ln.startswith("data:")][-1] == "data: [DONE]"
        with TestClient(create_lb_app(["http://127.0.0.1:9"])) as client:
            r = client.post("/v1/completions", json={"prompt": "x"})
            assert r.status_code == 502 and "replica 0 unavailable" in r.json()["error"]["message"]


def test_replica_and_tensor_parallel_launches(hw_a100, prepared_vllm, prepared_llamacpp, monkeypatch):
    import polyserve.backends.llamacpp as lc

    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/opt/llama-server")
    cfg = Config(backend="vllm", quant="fp8", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    rep = get_backend("vllm").replica_launch_spec(cfg, prepared_vllm, 9000, 1)
    assert rep.env["CUDA_VISIBLE_DEVICES"] == "1"
    lrep = get_backend("llamacpp-cuda").replica_launch_spec(
        Config(backend="llamacpp-cuda", quant="Q4_K_M", batch=4, n_gpu_layers=29), prepared_llamacpp, 9001, 0)
    assert lrep.env["CUDA_VISIBLE_DEVICES"] == "0"
    tp = get_backend("vllm").launch_spec(cfg.model_copy(update={"tp": 2}), prepared_vllm, 1).args
    assert tp[tp.index("--tensor-parallel-size") + 1] == "2"
    stp = get_backend("sglang").launch_spec(Config(backend="sglang", quant="bf16", tp=2), prepared_vllm, 1).args
    assert stp[stp.index("--tp-size") + 1] == "2"
    assert get_backend("vllm").supports_tp and not get_backend("llamacpp-cuda").supports_tp
    mm = get_backend("vllm").memory_model(hw_a100)
    one = estimate(hw_a100, prepared_vllm, cfg, mm)
    two = estimate(hw_a100, prepared_vllm, cfg.model_copy(update={"tp": 2}), mm)
    assert two.weights == one.weights // 2 and two.kv_cache == one.kv_cache // 2


def test_replica_runner_measures_through_the_balancer(tmp_path):
    from polyserve.layout import LayoutTrialRunner

    class Counting(FakeBackend):
        def __init__(self):
            super().__init__()
            self.pinned: List[str] = []

        def replica_launch_spec(self, cfg, model, port, gpu_index):
            spec = super().replica_launch_spec(cfg, model, port, gpu_index)
            self.pinned.append(spec.env["CUDA_VISIBLE_DEVICES"])
            return spec

    be = Counting()
    wl = Workload(n_prompts=4, prefill_tokens=8, decode_tokens=4, concurrencies=(2,))
    res = LayoutTrialRunner(be, None, make_hw("cpu"), wl, log_dir=tmp_path, startup_timeout=15).run_replicas(
        Config(backend="fake", quant="none", ctx=128, batch=1), [0, 1], "layout")
    assert res.ok, res.error
    assert res.replicas == 2 and res.metrics.output_tokens == 16 and be.pinned == ["0", "1"]


def test_replica_supervisor_serves_and_stops(tmp_path):
    from polyserve.layout import ReplicaSupervisor, create_lb_app

    sup = ReplicaSupervisor(FakeBackend(), Config(backend="fake", quant="none", ctx=128, batch=1), None, [0, 1],
                            log_dir=tmp_path, startup_timeout=15)
    sup.start()
    try:
        assert sup.healthy() and len(set(sup.urls)) == 2
        st = sup.status()
        assert st["layout"] == "replicas" and [r["gpu"] for r in st["replicas"]] == [0, 1]
        with TestClient(create_lb_app(sup.urls, status_fn=sup.status)) as client:
            r = client.post("/v1/completions", json={"model": "fake", "prompt": "hi", "max_tokens": 2})
            assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 2
    finally:
        sup.stop()
    assert not any(s.process.alive() for s in sup.supervisors)


class FakeReplicaRunner:
    def __init__(self, tok_s: float):
        self.tok_s = tok_s

    def run_replicas(self, cfg, gpus, stage):
        m = TrialMetrics(tok_s=self.tok_s, ttft_ms=40, tpot_ms=6, requests=48, output_tokens=6000)
        return TrialResult(config=cfg, stage=stage, metrics=m, replicas=len(gpus))


class FakeTPRunner:
    def __init__(self, tok_s: float):
        self.tok_s = tok_s
        self.calls: List[str] = []

    def run(self, cfg, stage):
        self.calls.append(cfg.key())
        m = TrialMetrics(tok_s=self.tok_s, ttft_ms=40, tpot_ms=6, requests=48, output_tokens=6000)
        return TrialResult(config=cfg, stage=stage, metrics=m)


@pytest.fixture
def single_gpu_winner(no_network, spec):
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    hw = two_gpus()
    candidates, reg = select(hw, spec, force="vllm")
    base = calibrate(hw, spec, "throughput", prepare_and_plan(hw, spec, candidates, reg), reg, runner=FakeRunner())
    return hw, reg, base


def test_layout_auto_keeps_the_best_arrangement(single_gpu_winner):
    from polyserve.layout import _single_reference, calibrate_layout

    hw, reg, base = single_gpu_winner
    ref = _single_reference(base).metrics.tok_s
    tp = FakeTPRunner(ref * 1.3)
    p = calibrate_layout(hw, "throughput", base, reg, "auto", replica_runner=FakeReplicaRunner(ref * 1.9),
                         tp_runner=tp)
    assert p.replicas == 2 and p.layout == "auto" and p.options.get("layout") == "auto"
    assert len(tp.calls) == 2 and all("tp2" in k for k in tp.calls)  # sharded, and sharded with a doubled batch
    assert any("2 replicas wins" in n for n in p.notes)
    tp_best = calibrate_layout(hw, "throughput", base, reg, "auto", replica_runner=FakeReplicaRunner(ref * 0.9),
                               tp_runner=FakeTPRunner(ref * 1.4))
    assert tp_best.replicas == 1 and tp_best.config.tp == 2 and "--tensor-parallel-size" in tp_best.launch_args
    neither = calibrate_layout(hw, "throughput", base, reg, "auto", replica_runner=FakeReplicaRunner(ref * 0.5),
                               tp_runner=FakeTPRunner(ref * 0.5))
    assert neither.replicas == 1 and neither.config == base.config
    assert any("serving on one GPU" in n for n in neither.notes)


@pytest.mark.usefixtures("no_network")
def test_layout_refuses_or_falls_back_on_one_gpu(hw_a100, spec):
    from polyserve.layout import calibrate_layout
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    candidates, reg = select(hw_a100, spec, force="vllm")
    base = calibrate(hw_a100, spec, "throughput", prepare_and_plan(hw_a100, spec, candidates, reg), reg,
                     runner=FakeRunner())
    with pytest.raises(RuntimeError, match="two NVIDIA GPUs"):
        calibrate_layout(hw_a100, "throughput", base, reg, "replicas", replica_runner=FakeReplicaRunner(1))
    p = calibrate_layout(hw_a100, "throughput", base, reg, "auto", replica_runner=FakeReplicaRunner(1))
    assert p.replicas == 1 and any("two NVIDIA GPUs" in n for n in p.notes)


# =========================================================================== options, cache, CLI


def test_options_shape_the_cache_path(tmp_home, hw_a100, spec, prepared_vllm):
    from polyserve import cache
    from tests.test_cache_and_pipeline import _profile

    assert SearchOptions().key() == {}
    opts = SearchOptions(quants=["bf16", "fp8"], kv_quant=False, speculative=False)
    assert opts.key() == {"quant": "bf16+fp8", "kv": "off", "spec": "off"}
    p = _profile(hw_a100, spec, prepared_vllm).model_copy(update={"options": opts.key()})
    path = cache.save(p)
    assert path.name == "balanced-kv-off-quant-bf16+fp8-spec-off.json"
    assert cache.load(hw_a100, spec, "balanced", options=opts.key()) is not None
    assert cache.load(hw_a100, spec, "balanced") is None  # a restricted profile is never served as the default


def test_cli_validates_the_new_options():
    from polyserve.cli import app

    runner = CliRunner()
    for args in (["--kv-quant", "maybe"], ["--speculative", "yes"], ["--layout", "sideways"], ["--quant", "int3"]):
        r = runner.invoke(app, ["bench", "x/y", *args])
        assert r.exit_code != 0, args
    r = runner.invoke(app, ["workloads"])
    assert r.exit_code == 0 and "chat-system" in r.output and "5632" in r.output


def test_phases_and_layout_are_exclusive(spec, hw_a100):
    from polyserve.pipeline import resolve_profile

    with pytest.raises(RuntimeError, match="not both"):
        resolve_profile(spec, hw=hw_a100, phases="auto", layout="replicas")


def test_trial_keys_sanitise_into_log_filenames(tmp_path):
    from polyserve.calibrate.search import SubprocessTrialRunner

    cfg = Config(backend="fake", quant="none", ctx=128, batch=1, spec_decode="draft:org/model:4")
    runner = SubprocessTrialRunner(backends={"fake": FakeBackend()}, models={"fake": None}, hw=make_hw("cpu"),
                                   workload=Workload(n_prompts=1, prefill_tokens=8, decode_tokens=2,
                                                     concurrencies=(1,)), log_dir=tmp_path, startup_timeout=15)
    res = runner.run(cfg, "spec")
    assert res.ok, res.error
    logs = list(tmp_path.glob("*.log"))
    assert logs and all(":" not in p.name and "/" not in p.name for p in logs)
    time.sleep(0)  # keep the import used on every platform
