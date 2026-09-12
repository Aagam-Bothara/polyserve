"""Run a PolyServe router in its own process: the replica load balancer or the prefill/decode splitter.

Calibration measures multi-engine layouts through their router, with the load generator in the
measuring process. A router served from a thread of that process shares its GIL, and at a few
thousand streamed tokens per second the router, not the engines, becomes the bottleneck: on a pair
of A40s, two replicas measured slower than one GPU. A separate process removes the contention.

    python -m polyserve.router lb --port 8100 http://127.0.0.1:9001 http://127.0.0.1:9002
    python -m polyserve.router pd --port 8100 <prefill_url> <decode_url>
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m polyserve.router")
    ap.add_argument("kind", choices=("lb", "pd"))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=None, help="upstream request timeout, seconds")
    ap.add_argument("urls", nargs="+")
    args = ap.parse_args(argv)
    if args.kind == "lb":
        from polyserve.layout import create_lb_app

        app = create_lb_app(args.urls, request_timeout=args.timeout)
    else:
        if len(args.urls) != 2:
            ap.error("pd needs exactly two URLs: prefill, decode")
        from polyserve.disagg import create_pd_app

        app = create_pd_app(args.urls[0], args.urls[1], request_timeout=args.timeout)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
