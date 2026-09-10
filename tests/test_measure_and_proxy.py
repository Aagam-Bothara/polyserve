"""Drive the HTTP measurement path and the proxy against a fake OpenAI-compatible SSE server."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from polyserve.backends.base import LlmtraceHooks
from polyserve.calibrate.measure import RequestOutcome, _metrics_from, _parse_sse_line, run_trial
from polyserve.calibrate.tokens import TokenCounter, worst_source
from polyserve.calibrate.workload import Workload
from polyserve.serve.proxy import create_app


def fake_upstream(token_delay_s: float = 0.002) -> FastAPI:
    app = FastAPI()
    app.state.requests = []

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(req: Request):
        return await _complete(req, with_usage=True)

    @app.post("/v1/completions_nousage")
    async def completions_nousage(req: Request):
        return await _complete(req, with_usage=False)

    async def _complete(req: Request, with_usage: bool):
        body = await req.json()
        app.state.requests.append(body)
        n = int(body.get("max_tokens", 8))
        if not body.get("stream"):
            return JSONResponse({"choices": [{"text": "x " * n}], "usage": {"completion_tokens": n}})

        async def gen():
            # Two tokens per chunk on purpose: chunk counting must not be mistaken for token counting.
            for i in range(0, n, 2):
                await asyncio.sleep(token_delay_s)
                yield f"data: {json.dumps({'choices': [{'text': 'x y '}]})}\n\n"
            if with_usage:
                yield f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


class _Server:
    def __init__(self, app: FastAPI):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        assert self.server.started
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture(scope="module")
def upstream():
    app = fake_upstream()
    with _Server(app) as srv:
        yield srv, app


def test_parse_sse_line():
    assert _parse_sse_line("data: [DONE]") is None
    assert _parse_sse_line(": comment") is None
    assert _parse_sse_line('data: {"a": 1}') == {"a": 1}


def test_metrics_from_outcomes():
    outs = [RequestOutcome(ok=True, ttft_s=0.1, duration_s=1.1, tokens=101),
            RequestOutcome(ok=True, ttft_s=0.3, duration_s=1.3, tokens=101),
            RequestOutcome(ok=False, error="x")]
    m = _metrics_from(outs, wall_s=2.0, concurrency=2)
    assert m.requests == 3 and m.failed == 1 and m.output_tokens == 202
    assert m.tok_s == 101.0
    assert m.ttft_ms == pytest.approx(200.0)
    assert m.tpot_ms == pytest.approx(10.0)
    assert not m.ok


def test_run_trial_against_fake_server(upstream):
    srv, app = upstream
    wl = Workload(n_prompts=4, prefill_tokens=32, decode_tokens=8, concurrencies=(1, 4))
    hooks = LlmtraceHooks(model_name="fake-model", gpu_ids=[])
    m = run_trial(srv.url, hooks, wl, pid=None, warmup=True)
    assert m.ok and m.requests == 8 and m.failed == 0
    assert m.output_tokens == 64 and m.token_count_source == "usage"  # usage.completion_tokens honoured
    assert set(m.by_concurrency) == {"1", "4"}
    assert m.by_concurrency["4"].tok_s >= m.by_concurrency["1"].tok_s * 0.8
    assert 0 < m.ttft_ms < 5000 and m.tpot_ms > 0
    assert app.state.requests[-1]["stream"] is True and app.state.requests[-1]["max_tokens"] == 8


def test_run_trial_reports_failures(upstream):
    srv, _ = upstream
    hooks = LlmtraceHooks(model_name="fake-model", completions_path="/v1/nope")
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    m = run_trial(srv.url, hooks, wl, warmup=False)
    assert m.failed == 2 and not m.ok


def test_proxy_streams_and_forwards(upstream):
    srv, _ = upstream
    app = create_app(srv.url, profile=None, status_fn=lambda: {"alive": True})
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        r = client.get("/v1/models")
        assert r.json()["data"][0]["id"] == "fake-model"
        r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
        assert r.json()["usage"]["completion_tokens"] == 3
        with client.stream("POST", "/v1/completions", json={"prompt": "hi", "max_tokens": 3, "stream": True}) as s:
            assert s.headers["content-type"].startswith("text/event-stream")
            lines = [ln for ln in s.iter_lines() if ln.startswith("data:")]
        assert lines[-1] == "data: [DONE]" and len(lines) == 4  # 2 content chunks + usage + DONE
        assert client.get("/polyserve/profile").status_code == 404


def test_proxy_reports_upstream_down():
    app = create_app("http://127.0.0.1:9", profile=None, status_fn=lambda: {"alive": False})
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        r = client.post("/v1/completions", json={"prompt": "hi"})
        assert r.status_code == 502 and "backend unavailable" in r.json()["error"]["message"]


def test_proxy_profile_endpoint(hw_a100, spec, prepared_vllm):
    from tests.test_cache_and_pipeline import _profile

    p = _profile(hw_a100, spec, prepared_vllm)
    app = create_app("http://127.0.0.1:9", profile=p, status_fn=lambda: {"alive": True, "restarts": 0})
    with TestClient(app) as client:
        data = client.get("/polyserve/profile").json()
        assert data["backend"] == "vllm" and data["config"]["batch"] == 64
        assert data["runtime"]["restarts"] == 0


def test_telemetry_without_gpu_falls_back_to_process(monkeypatch):
    import os

    from polyserve.calibrate.measure import Telemetry

    with Telemetry(LlmtraceHooks(gpu_ids=[], process_memory=True), pid=os.getpid(), interval_ms=20) as t:
        time.sleep(0.15)
    assert t.summary.source in ("psutil", "rapl")
    if t.summary.source == "psutil":
        assert t.summary.peak_mem_mb and t.summary.peak_mem_mb > 1


def _word_counter() -> TokenCounter:
    return TokenCounter(encode=lambda text: len(text.split()), name="fake-words")


def test_tokens_from_tokenizer_when_usage_absent(upstream):
    srv, _ = upstream
    hooks = LlmtraceHooks(model_name="fake-model", completions_path="/v1/completions_nousage")
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=8, concurrencies=(1,))
    m = run_trial(srv.url, hooks, wl, warmup=False, counter=_word_counter())
    # 8 tokens streamed as 4 chunks of "x y ": the tokenizer sees 8 words, chunk counting would say 4.
    assert m.output_tokens == 16 and m.token_count_source == "tokenizer"
    assert abs(m.prompt_tokens - 8) <= 2 and wl.fitted  # fitter tolerance; fixed prefix+suffix set a floor


def test_tokens_from_chunks_is_flagged_when_nothing_better(upstream):
    srv, _ = upstream
    hooks = LlmtraceHooks(model_name="fake-model", completions_path="/v1/completions_nousage")
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=8, concurrencies=(1,))
    m = run_trial(srv.url, hooks, wl, warmup=False, counter=TokenCounter())
    assert m.output_tokens == 8 and m.token_count_source == "chunks"  # 4 chunks x 2 requests, approximate
    assert m.prompt_tokens == 0 and not wl.fitted


def test_worst_source_and_prompt_fitting():
    assert worst_source(["usage", "usage"]) == "usage"
    assert worst_source(["usage", "tokenizer"]) == "tokenizer"
    assert worst_source(["tokenizer", "chunks", "usage"]) == "chunks"
    assert worst_source([]) == "none"
    wl = Workload(n_prompts=3, prefill_tokens=100, decode_tokens=4).fit_prompts(_word_counter())
    assert wl.fitted and all(abs(len(p.split()) - 100) <= 2 for p in wl.prompts)
    assert "~" not in wl.describe()
