"""Thin OpenAI-compatible proxy in front of the supervised backend.

Exposes /v1/chat/completions, /v1/completions, /v1/models (and any other /v1/* the backend
serves), plus /polyserve/profile and /health. Streaming responses are passed through byte-for-byte.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from polyserve.models import Profile

logger = logging.getLogger(__name__)

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length", "content-encoding", "host",
}


def _filtered(headers: httpx.Headers) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def _ask_for_usage(raw: bytes) -> bytes:
    """Add `stream_options: {"include_usage": true}` to a streaming request, leaving everything else alone.

    A streamed reply carries no token counts unless it was asked for, and most OpenAI clients stream, so without
    this the drift report would only ever see the non-streaming minority. The extra final chunk carries usage and
    an empty `choices`, which the OpenAI clients accept; a request that already set stream_options is untouched.
    """
    if not raw:
        return raw
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(body, dict) or not body.get("stream"):
        return raw
    options = body.get("stream_options")
    if isinstance(options, dict) and "include_usage" in options:
        return raw
    body["stream_options"] = {**(options if isinstance(options, dict) else {}), "include_usage": True}
    try:
        return json.dumps(body).encode()
    except (TypeError, ValueError):
        return raw


def create_app(
    upstream_base_url: str,
    profile: Optional[Profile] = None,
    status_fn=None,
    request_timeout: Optional[float] = None,
    watch=None,
) -> FastAPI:
    state: Dict[str, Any] = {"client": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["client"] = httpx.AsyncClient(base_url=upstream_base_url, timeout=request_timeout)
        try:
            yield
        finally:
            await state["client"].aclose()

    app = FastAPI(title="PolyServe", version=profile.polyserve_version if profile else "dev", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        body: Dict[str, Any] = {"status": "ok", "upstream": upstream_base_url}
        if status_fn is not None:
            body["backend"] = status_fn()
            if not body["backend"].get("alive", True):
                body["status"] = "degraded"
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.get("/polyserve/profile")
    async def get_profile() -> JSONResponse:
        if profile is None:
            return JSONResponse({"error": "no profile (running with defaults)"}, status_code=404)
        data = profile.model_dump(mode="json")
        # Keep the wire payload small: the full hardware descriptor and trial table are large.
        data["calibration_table"] = [
            {
                "stage": t["stage"],
                "config": t["config"],
                "ok": t["error"] is None and t["launched"],
                "tok_s": t["metrics"]["tok_s"],
                "ttft_ms": t["metrics"]["ttft_ms"],
                "tpot_ms": t["metrics"]["tpot_ms"],
                "peak_mem_mb": t["metrics"]["peak_mem_mb"],
                "power_w": t["metrics"]["power_w"],
                "joules_per_token": t["metrics"]["joules_per_token"],
                "error": t["error"],
            }
            for t in data.get("calibration_table", [])
        ]
        if status_fn is not None:
            data["runtime"] = status_fn()
        return JSONResponse(data)

    @app.get("/polyserve/drift")
    async def drift() -> JSONResponse:
        """Whether live traffic still looks like the workload this profile was calibrated on."""
        if watch is None:
            return JSONResponse({"error": "no traffic watch (serving without a profile)"}, status_code=404)
        return JSONResponse(watch.report())

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
    async def passthrough(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = state["client"]
        body = await request.body()
        if watch is not None:
            body = _ask_for_usage(body)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        upstream_req = client.build_request(
            request.method,
            f"/v1/{path}",
            content=body,
            headers=headers,
            params=request.query_params,
        )
        if watch is not None:
            watch.began()
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("upstream error: %s", exc)
            if watch is not None:
                watch.ended()
            return JSONResponse({"error": {"message": f"backend unavailable: {exc}", "type": "upstream_error"}},
                                status_code=502)
        media = upstream.headers.get("content-type", "")
        if "text/event-stream" in media:
            # Streamed bytes are never parsed, so a streamed reply counts as a request and toward concurrency,
            # but carries no token counts; the request is only over once the client has drained it.
            async def _finish_stream() -> None:
                await upstream.aclose()
                if watch is not None:
                    watch.ended()

            async def _tee():
                """Pass the bytes through untouched, reading the usage chunk on the way past."""
                async for chunk in upstream.aiter_raw():
                    if watch is not None:
                        watch.record_stream_chunk(chunk)
                    yield chunk

            return StreamingResponse(
                _tee(),
                status_code=upstream.status_code,
                headers=_filtered(upstream.headers),
                media_type="text/event-stream",
                background=BackgroundTask(_finish_stream),
            )
        content = await upstream.aread()
        await upstream.aclose()
        if watch is not None:
            watch.record_response(content, media)
            watch.ended()
            if watch.should_warn():
                logger.warning("traffic no longer looks like the calibrated workload: %s; "
                               "`polyserve recalibrate` would re-tune for it", "; ".join(watch.findings()))
        return Response(content=content, status_code=upstream.status_code, headers=_filtered(upstream.headers))

    return app
