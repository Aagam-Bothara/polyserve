"""Held-out prompts from OpenAssistant: the opening user turn of English conversations, deduplicated."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_oasst_prompts", Path(__file__).resolve().parents[1] / "benchmarks" / "make_oasst_prompts.py")
mop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mop)

ROWS = [
    {"text": "How do tides work?", "role": "prompter", "parent_id": None, "lang": "en"},
    {"text": "The Moon pulls the oceans.", "role": "assistant", "parent_id": "m1", "lang": "en"},  # a reply
    {"text": "And why twice a day?", "role": "prompter", "parent_id": "m2", "lang": "en"},  # a later turn
    {"text": "¿Qué es la marea?", "role": "prompter", "parent_id": None, "lang": "es"},  # another language
    {"text": " How do tides work? ", "role": "prompter", "parent_id": None, "lang": "en"},  # a duplicate opener
    {"text": "Write a haiku about rain.", "role": "prompter", "parent_id": None, "lang": "en"},
]


def test_only_distinct_english_openers_are_kept():
    assert sorted(mop.first_turns(ROWS, 10)) == ["How do tides work?", "Write a haiku about rain."]


def test_the_selection_is_seeded_and_capped():
    assert mop.first_turns(ROWS, 1) == mop.first_turns(ROWS, 1) and len(mop.first_turns(ROWS, 1)) == 1
