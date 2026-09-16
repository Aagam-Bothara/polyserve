from __future__ import annotations

import json

from fastapi.testclient import TestClient

from polyserve.drift import TrafficWatch
from polyserve.serve.proxy import _ask_for_usage, create_app

CHAT = {"name": "chat", "prefill_tokens": 512, "decode_tokens": 128, "concurrencies": [1, 4, 8]}


def _body(prompt: int, completion: int) -> bytes:
    return json.dumps({"choices": [{"text": "x"}],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}).encode()


def _serve(watch: TrafficWatch, n: int, prompt: int = 500, completion: int = 120, concurrent: int = 1) -> None:
    """n requests, `concurrent` of them in flight at a time, each answered with a usage block."""
    for _ in range(0, n, concurrent):
        for _ in range(concurrent):
            watch.began()
        for _ in range(concurrent):
            watch.record_response(_body(prompt, completion), "application/json")
            watch.ended()


def test_traffic_like_the_calibrated_workload_reports_no_drift():
    watch = TrafficWatch(CHAT)
    _serve(watch, 60)

    report = watch.report()
    assert report["drift"] == []
    assert report["verdict"] == "matches the calibrated workload"
    assert report["requests_seen"] == 60 and report["requests_with_token_counts"] == 60


def test_longer_prompts_are_reported_as_drift():
    watch = TrafficWatch(CHAT)
    _serve(watch, 60, prompt=3000)

    findings = watch.findings()
    assert len(findings) == 1 and "prompts are" in findings[0] and "longer" in findings[0]
    assert "3000" in findings[0] and "512" in findings[0]
    assert watch.report()["verdict"] == "drifted from the calibrated workload"


def test_much_shorter_answers_are_reported_as_drift():
    watch = TrafficWatch(CHAT)
    _serve(watch, 60, completion=8)

    findings = watch.findings()
    assert len(findings) == 1 and "answers are" in findings[0] and "shorter" in findings[0]


def test_concurrency_above_every_calibrated_level_is_reported():
    watch = TrafficWatch(CHAT)
    _serve(watch, 64, concurrent=32)

    findings = watch.findings()
    # The sentence reports typical load, which is the median of a ramp to 32, not its peak.
    assert any("concurrent requests" in f and "1/4/8" in f for f in findings)
    assert watch.report()["concurrency"]["peak"] == 32
    assert watch.report()["concurrency"]["median"] > 8


def test_traffic_below_the_minimum_is_not_called_drift():
    watch = TrafficWatch(CHAT)
    _serve(watch, 10, prompt=9000)

    assert watch.findings() == [] and watch.report()["verdict"] == "not enough traffic yet"
    assert watch.report()["enough_data"] is False


def test_streamed_replies_count_as_requests_without_token_counts():
    watch = TrafficWatch(CHAT)
    for _ in range(60):
        watch.began()
        watch.record_response(b"data: {}\n\n", "text/event-stream")
        watch.ended()

    report = watch.report()
    assert report["requests_seen"] == 60 and report["requests_with_token_counts"] == 0
    assert report["prompt_tokens"]["median"] is None and report["drift"] == []


def test_the_operator_is_warned_once():
    watch = TrafficWatch(CHAT)
    _serve(watch, 60, prompt=3000)

    assert watch.should_warn() is True
    assert watch.should_warn() is False


def test_a_malformed_body_is_ignored():
    watch = TrafficWatch(CHAT)
    watch.began()
    watch.record_response(b"not json at all", "application/json")
    watch.ended()

    assert watch.report()["requests_with_token_counts"] == 0


def test_the_proxy_exposes_the_report_and_balances_in_flight():
    watch = TrafficWatch(CHAT)
    app = create_app("http://127.0.0.1:9", profile=None, status_fn=lambda: {"alive": False}, watch=watch)

    with TestClient(app) as client:
        assert client.post("/v1/completions", json={"prompt": "hi"}).status_code == 502
        report = client.get("/polyserve/drift").json()

    assert report["requests_seen"] == 1
    assert report["calibrated_workload"] == "chat"
    assert watch.in_flight == 0  # a failed upstream still ends the request


def test_the_drift_endpoint_is_absent_without_a_watch():
    app = create_app("http://127.0.0.1:9", profile=None, status_fn=lambda: {"alive": False})

    with TestClient(app) as client:
        assert client.get("/polyserve/drift").status_code == 404


def test_streamed_requests_are_asked_for_their_usage():
    """Most OpenAI clients stream, and a streamed reply carries no token counts unless the request asks."""
    out = json.loads(_ask_for_usage(json.dumps({"prompt": "hi", "stream": True}).encode()))

    assert out["stream_options"] == {"include_usage": True}
    assert out["prompt"] == "hi" and out["stream"] is True


def test_requests_that_should_not_be_touched_are_not():
    plain = json.dumps({"prompt": "hi"}).encode()  # not streaming: nothing to ask for
    assert _ask_for_usage(plain) == plain
    assert _ask_for_usage(b"not json at all") == b"not json at all"
    assert _ask_for_usage(b"") == b""
    chosen = json.dumps({"prompt": "hi", "stream": True, "stream_options": {"include_usage": False}}).encode()
    assert _ask_for_usage(chosen) == chosen  # a caller who already decided keeps their decision


def test_token_counts_are_taken_from_a_streamed_usage_chunk():
    watch = TrafficWatch(CHAT)
    watch.began()
    watch.record_stream_chunk(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
    watch.record_stream_chunk(
        b'data: {"choices":[],"usage":{"prompt_tokens":700,"completion_tokens":120}}\n\ndata: [DONE]\n\n')
    watch.ended()

    report = watch.report()
    assert report["requests_with_token_counts"] == 1
    assert report["prompt_tokens"]["median"] == 700 and report["completion_tokens"]["median"] == 120
