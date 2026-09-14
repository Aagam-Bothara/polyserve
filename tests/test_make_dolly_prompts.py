"""The Dolly prompt file for the --workload-file results: context first, then the instruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_dolly_prompts", Path(__file__).resolve().parents[1] / "benchmarks" / "make_dolly_prompts.py")
mdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdp)


def test_a_row_with_context_puts_the_passage_before_the_instruction():
    row = {"instruction": " Summarise the passage. ", "context": "Some passage.\n"}
    assert mdp.to_prompt(row) == "Some passage.\n\nSummarise the passage."


def test_a_row_without_context_is_just_the_instruction():
    assert mdp.to_prompt({"instruction": "Name three rivers.", "context": ""}) == "Name three rivers."
    assert mdp.to_prompt({"instruction": "Name three rivers."}) == "Name three rivers."
