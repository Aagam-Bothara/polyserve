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


FILE_PREFIX = "file:"  # a source naming your own prompt file: "file:/path/to/prompts.jsonl"


def read_prompt_file(path) -> List[str]:
    """Your own prompts, from a JSONL file with one per line: a JSON string, {"prompt": ...}, {"text": ...}
    or {"messages": [{"role": ..., "content": ...}, ...]}. A conversation's messages are joined in order with
    blank lines; no chat template is applied, as with the built-in sources, which send raw text too."""
    out: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not valid JSON ({exc.msg})") from None
            text = _prompt_text(item)
            if not text:
                raise ValueError(f'{path}:{n}: expected a string, or an object with "prompt", "text" or "messages"')
            out.append(text)
    if not out:
        raise ValueError(f"{path}: no prompts")
    return out


def _prompt_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("prompt", "text"):
            if isinstance(item.get(key), str):
                return item[key].strip()
        messages = item.get("messages")
        if isinstance(messages, list):
            return "\n\n".join(str(m["content"]).strip() for m in messages
                               if isinstance(m, dict) and m.get("content")).strip()
    return ""


@functools.lru_cache(maxsize=None)
def pool(source: str) -> Tuple[Tuple[str, str], ...]:
    """Every usable prompt of a source, in one fixed shuffled order."""
    if source.startswith(FILE_PREFIX):
        items = [(p, "") for p in read_prompt_file(source[len(FILE_PREFIX):])]
    elif source not in LOADERS:
        raise ValueError(f"unknown prompt source {source!r}; choose from {', '.join(LOADERS)} or file:<path>")
    else:
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
