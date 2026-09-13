"""Real prompts for the dataset-backed workloads.

Random-word prompts measure throughput well, but not strategies that depend on what the text says:
n-gram speculative decoding only pays when an answer repeats its input, and how long an answer runs
depends on the question. These sources supply real text:

  sharegpt        first user turns of real ChatGPT conversations (ShareGPT, as vLLM's benchmarks use)
  cnn-extract     news articles, with an instruction to quote sentences word for word (answers copy input)
  humaneval-edit  Python functions, with an instruction to add type hints (answers copy the code)

Each source is downloaded from the Hugging Face Hub on first use and cached there.
"""

from __future__ import annotations

import functools
import json
import random
from typing import Callable, Dict, List, Tuple

EXTRACT_INSTRUCTION = ("\n\nCopy, word for word, every sentence in the article above that contains a number. "
                       "Output only those sentences, one per line.")
EDIT_INSTRUCTION = ("\n\nRewrite the Python function above with type hints on every parameter and on the return "
                    "value, and a one-line docstring. Keep every other line exactly as it is. Output only the code.")
CHARS_PER_TOKEN = 4  # rough English average, for a first cut before a tokenizer trims precisely


def _sharegpt() -> List[Tuple[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                           repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for conv in data:
        turns = conv.get("conversations") or []
        if turns and turns[0].get("from") == "human" and len((turns[0].get("value") or "").strip()) >= 20:
            out.append((turns[0]["value"].strip(), ""))
    return out


def _cnn_extract() -> List[Tuple[str, str]]:
    from datasets import load_dataset

    return [(row["article"].strip(), EXTRACT_INSTRUCTION)
            for row in load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")]


def _humaneval_edit() -> List[Tuple[str, str]]:
    from datasets import load_dataset

    return [((row["prompt"] + row["canonical_solution"]).rstrip(), EDIT_INSTRUCTION)
            for row in load_dataset("openai/openai_humaneval", split="test")]


# source -> loader of (body, instruction) pairs; the instruction survives any trimming of the body.
LOADERS: Dict[str, Callable[[], List[Tuple[str, str]]]] = {
    "sharegpt": _sharegpt, "cnn-extract": _cnn_extract, "humaneval-edit": _humaneval_edit,
}


@functools.lru_cache(maxsize=None)
def pool(source: str) -> Tuple[Tuple[str, str], ...]:
    """Every usable prompt of a source, in one fixed shuffled order."""
    if source not in LOADERS:
        raise ValueError(f"unknown prompt source {source!r}; choose from {', '.join(LOADERS)}")
    items = list(LOADERS[source]())
    if not items:
        raise RuntimeError(f"prompt source {source!r} returned nothing")
    random.Random(0).shuffle(items)
    return tuple(items)


def sample(source: str, n: int, offset: int, max_chars: int) -> List[str]:
    """n prompts from position `offset` of the shuffled pool (wrapping around a small one), each body
    cut to fit max_chars with its instruction, at a line break when there is one nearby."""
    items = pool(source)
    out = []
    for j in range(n):
        body, instruction = items[(offset + j) % len(items)]
        room = max(64, max_chars - len(instruction))
        if len(body) > room:
            cut = body.rfind("\n", 0, room)
            body = body[:cut] if cut > room // 2 else body[:room]
        out.append(body + instruction)
    return out
