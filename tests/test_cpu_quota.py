"""Inside a container PolyServe sizes CPU threads to the cgroup quota, not the host's visible cores."""

from __future__ import annotations

import polyserve.hardware as hwmod


def test_cgroup_v2_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("765000 100000\n")
    assert hwmod._cpu_quota(str(tmp_path)) == 7.65
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert hwmod._cpu_quota(str(tmp_path)) is None


def test_cgroup_v1_quota(tmp_path):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("200000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert hwmod._cpu_quota(str(tmp_path)) == 2.0
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    assert hwmod._cpu_quota(str(tmp_path)) is None
    assert hwmod._cpu_quota(str(tmp_path / "missing")) is None


def test_probe_caps_cores_to_the_usable_cpus(monkeypatch):
    monkeypatch.setattr(hwmod.psutil, "cpu_count", lambda logical=True: 96 if logical else 48)
    monkeypatch.setattr(hwmod, "_usable_cpus", lambda: 7)  # RunPod A40 pod: 96 visible, 7.65-CPU quota
    c = hwmod.probe_cpu()
    assert (c.physical_cores, c.logical_cores) == (7, 7)
    monkeypatch.setattr(hwmod, "_usable_cpus", lambda: None)  # no container limit: unchanged
    c = hwmod.probe_cpu()
    assert (c.physical_cores, c.logical_cores) == (48, 96)
