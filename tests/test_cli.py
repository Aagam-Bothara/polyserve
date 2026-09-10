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
