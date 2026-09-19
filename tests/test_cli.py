from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from polyserve import __version__
from polyserve.cli import app

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_probe_json():
    r = runner.invoke(app, ["probe", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "hardware_hash" in data and "backends" in data


def test_profiles_empty(tmp_home):
    r = runner.invoke(app, ["profiles"])
    assert r.exit_code == 0 and "no profiles" in r.output


def test_bad_objective():
    r = runner.invoke(app, ["bench", "x/y", "--objective", "fast"])
    assert r.exit_code != 0


@pytest.mark.usefixtures("no_network")
def test_plan_command(hw_a100, monkeypatch):
    import polyserve.hardware as hwmod

    monkeypatch.setattr(hwmod, "probe", lambda: hw_a100)
    r = runner.invoke(app, ["plan", "meta-llama/Llama-3.2-3B-Instruct", "--backend", "vllm", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["vllm"] and data["_errors"] == {}


def test_serve_listens_on_this_machine_only_by_default():
    """0.0.0.0 would put an unauthenticated inference endpoint on every network the host can reach."""
    import typer.main

    serve = typer.main.get_command(app).commands["serve"]
    host = next(p for p in serve.params if p.name == "host")
    assert host.default == "127.0.0.1"
    assert host.envvar == "POLYSERVE_HOST"  # how the Docker image opts back in


def test_loopback_detection():
    from polyserve.cli import _is_loopback

    assert _is_loopback("127.0.0.1") and _is_loopback("localhost") and _is_loopback("::1")
    assert _is_loopback("127.0.0.2")  # the whole 127/8 block is loopback
    assert not _is_loopback("0.0.0.0") and not _is_loopback("10.0.0.5") and not _is_loopback("my-gpu-box")


def test_the_docker_image_opts_back_in_to_every_interface():
    """Inside a container, -p 8000:8000 only reaches a server listening on 0.0.0.0. Without this line the
    documented `docker run` would start cleanly and be unreachable, and no other test would notice."""
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV POLYSERVE_HOST=0.0.0.0" in dockerfile
