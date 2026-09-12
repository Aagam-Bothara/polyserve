"""Stopping a backend reaps its whole process group, even after the server itself has exited."""

from __future__ import annotations

import os
import sys
import time

import pytest

from polyserve.backends.base import LaunchSpec, Process


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")
def test_stop_reaps_children_the_server_left_behind(tmp_path):
    # The "server" starts a long-lived child (an engine core, say) and then dies on its own.
    pidfile = tmp_path / "child.pid"
    spec = LaunchSpec(args=["sh", "-c", f"sleep 300 & echo $! > {pidfile}; exit 1"])
    proc = Process(spec, port=1, health_url="http://127.0.0.1:1/health", log_path=tmp_path / "server.log").start()
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip() and not proc.alive():
            break
        time.sleep(0.05)
    child = int(pidfile.read_text())
    os.kill(child, 0)  # still running although its parent is gone
    proc.stop()
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail("the orphaned child survived Process.stop()")
