"""--all-levels switches off the busiest-level-first shortcut, so the shortcut itself can be checked."""

from __future__ import annotations

from typer.testing import CliRunner

from polyserve.calibrate.objectives import Constraints
from polyserve.cli import _opts, app
from polyserve.pipeline import SearchOptions, level_rule


def test_the_shortcut_is_on_by_default_and_off_when_asked():
    cons = Constraints(ttft_ceiling_ms=500)
    assert level_rule("balanced", cons, SearchOptions()) is not None
    assert level_rule("balanced", cons, SearchOptions(all_levels=True)) is None
    assert level_rule("balanced", cons, SearchOptions(), power_points=["a power point"]) is None
    assert level_rule("latency", cons, SearchOptions()) is None  # latency can prefer a quieter level anyway


def test_a_full_level_profile_is_cached_apart():
    assert SearchOptions(all_levels=True).key() == {"levels": "all"} and SearchOptions().key() == {}
    assert _opts("auto", "on", "on", "on", all_levels=True).all_levels
    assert not _opts("auto", "on", "on", "on").all_levels


def test_every_calibrating_command_takes_the_flag():
    for command in ("bench", "recalibrate", "compare", "serve"):
        assert "--all-levels" in CliRunner().invoke(app, [command, "--help"], terminal_width=200).output
