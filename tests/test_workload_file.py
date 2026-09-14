"""--workload-file: calibrate on your own prompts, shaped like a preset."""

from __future__ import annotations

import json
import logging

import pytest

from polyserve.calibrate import datasets
from polyserve.calibrate.measure import _level_workload
from polyserve.calibrate.tokens import TokenCounter
from polyserve.calibrate.workload import get_workload, workload_from_file


def _write(tmp_path, lines, name="prompts.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_prompt_file_formats(tmp_path):
    p = _write(tmp_path, ['"plain string"', '{"prompt": "a prompt"}', "", '{"text": "a text"}',
                          '{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}]}'])
    assert datasets.read_prompt_file(p) == ["plain string", "a prompt", "a text", "Be brief.\n\nHi?"]


def test_prompt_file_errors_name_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":2: not valid JSON"):
        datasets.read_prompt_file(_write(tmp_path, ['{"prompt": "ok"}', "{broken"]))
    with pytest.raises(ValueError, match=r":1: expected"):
        datasets.read_prompt_file(_write(tmp_path, ['{"answer": "x"}'], "b.jsonl"))
    with pytest.raises(ValueError, match="no prompts"):
        datasets.read_prompt_file(_write(tmp_path, [""], "c.jsonl"))


def test_the_template_sets_concurrency_and_ceilings(tmp_path):
    p = _write(tmp_path, [json.dumps({"prompt": f"question {i} " + "word " * 100}) for i in range(60)])
    wl, chat = workload_from_file(p, get_workload("chat")), get_workload("chat")
    assert (wl.concurrencies, wl.ttft_ceiling_ms, wl.tpot_ceiling_ms) == (chat.concurrencies, chat.ttft_ceiling_ms,
                                                                          chat.tpot_ceiling_ms)
    assert wl.natural_stop and wl.decode_tokens == 512 and wl.name.startswith("file-prompts-")
    assert wl.source == datasets.FILE_PREFIX + str(p.resolve()) and wl.prefill_tokens >= 256
    assert len(wl.ensure_prompts().prompts) == wl.n_prompts and "prompts.jsonl" in wl.describe()


def test_editing_the_file_gives_a_new_workload_and_so_a_new_profile(tmp_path):
    p = _write(tmp_path, ['{"prompt": "one"}'])
    first = workload_from_file(p).name
    p.write_text('{"prompt": "two"}\n', encoding="utf-8")
    assert workload_from_file(p).name != first


def test_each_concurrency_level_gets_fresh_prompts(tmp_path):
    p = _write(tmp_path, [json.dumps({"prompt": f"distinct prompt {i}"}) for i in range(100)])
    wl = workload_from_file(p, get_workload("chat")).ensure_prompts()  # levels 1/4/8 at 4 per slot: 8, 16, 32
    counter, sent = TokenCounter(), []
    for k, c in enumerate(wl.concurrencies):
        level = _level_workload(wl, k, counter)
        sent.append(set(level.prompts[: level.level_requests(c)]))
    assert [len(s) for s in sent] == [8, 16, 32]
    assert all(not (a & b) for i, a in enumerate(sent) for b in sent[i + 1:])  # the prefix cache sees no repeats


def test_a_small_file_sends_fewer_requests_and_says_so(tmp_path, caplog):
    p = _write(tmp_path, [json.dumps({"prompt": f"p{i}"}) for i in range(12)])
    with caplog.at_level(logging.WARNING):
        wl = workload_from_file(p, get_workload("chat"))
    assert wl.n_prompts == 4  # 12 prompts over 3 levels, rather than reusing any
    assert "12 prompts" in caplog.text
