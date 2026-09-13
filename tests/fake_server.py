"""Minimal OpenAI-compatible SSE server used as a stand-in backend subprocess in tests.

    python tests/fake_server.py --port 8123 [--startup-delay 0.5] [--die-after 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CRASHABLE = False  # --crashable: GET /crash exits abruptly, so a test decides when the crash happens


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        if self.path == "/crash" and CRASHABLE:
            os._exit(9)
        if self.path == "/health":
            body = b'{"status":"ok"}'
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        tokens = int(req.get("max_tokens", 4))
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(tokens):
                self.wfile.write(f"data: {json.dumps({'choices': [{'text': 'x '}]})}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.001)
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': tokens}})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = json.dumps({"choices": [{"text": "x " * tokens}], "usage": {"completion_tokens": tokens}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--startup-delay", type=float, default=0.0)
    ap.add_argument("--die-after", type=float, default=0.0, help="exit abruptly after N seconds")
    ap.add_argument("--die-once", default="", help="marker path; die only on the first start, then stay up")
    ap.add_argument("--fail", action="store_true", help="exit 3 immediately")
    ap.add_argument("--crashable", action="store_true", help="GET /crash exits abruptly")
    args = ap.parse_args()
    global CRASHABLE
    CRASHABLE = args.crashable
    if args.fail:
        print("fake backend: refusing to start", flush=True)
        sys.exit(3)
    time.sleep(args.startup_delay)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    should_die = args.die_after > 0
    if should_die and args.die_once:
        # Crash exactly once so a supervisor's restart can be observed settling into a healthy state.
        if os.path.exists(args.die_once):
            should_die = False
        else:
            open(args.die_once, "w").close()
    if should_die:
        threading.Timer(args.die_after, lambda: os._exit(9)).start()
    print(f"fake backend listening on {args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
