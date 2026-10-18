import json
from pathlib import Path

import pytest

from syncytium.sweep.config import EvalConfig, ModelConfig
from syncytium.tasks import (
    GSM8KConfig,
    MMLUConfig,
    SwarmSetting,
    TaskConfig,
    TaskData,
    TaskRun,
    items_path,
    plan,
)
from syncytium.tasks import runner as runner_module
from syncytium.tasks.gsm8k import Problem
from syncytium.tasks.mmlu import Question

# The tiny model's narrowest input dimension is 32, so groups of 32 fit every weight.
SETTING = SwarmSetting(bits=4, group_size=32, children=2, phase=0.5)


def config(tmp_path: Path) -> TaskConfig:
    return TaskConfig(
        name="tiny",
        output=tmp_path,
        model=ModelConfig(path="unused"),
        evaluation=EvalConfig(),
        setting=SETTING,
        gsm8k=GSM8KConfig(split="train", batch_size=2, max_new_tokens=6),
        mmlu=MMLUConfig(split="dev", batch_size=2),
    )


@pytest.fixture
def data(tokenizer):
    problems = [Problem(f"question {i}", str(i)) for i in range(3)]
    questions = [Question("anatomy", f"q{i}", ("a", "b", "c", "d"), i % 4) for i in range(3)]
    return TaskData(
        problems,
        [tokenizer.encode(p.question) for p in problems],
        questions,
        [tokenizer.encode(q.question) for q in questions],
    )


def read(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_every_condition_is_logged_with_its_answers(tiny_model, tokenizer, data, tmp_path):
    conditions = plan(SETTING.children, drop=True)
    TaskRun(tiny_model, tokenizer, config(tmp_path), data).run(conditions)
    summary = {row["id"]: row for row in read(tmp_path / "results.jsonl")}
    non_drops = [c for c in conditions if c.kind != "drop"]
    expected = {f"gsm8k/{c.id}" for c in conditions} | {f"mmlu/{c.id}" for c in non_drops}
    assert set(summary) == expected
    for row in summary.values():
        assert row["total"] == 3
        assert 0 <= row["accuracy"] <= 1
    answers = read(items_path(tmp_path, "gsm8k", "group/k0,1"))
    assert [a["index"] for a in answers] == [0, 1, 2]
    assert all(len(a["tokens"]) <= 6 for a in answers)


def test_a_drop_keeps_the_first_half_of_the_full_group_answer(
    tiny_model, tokenizer, data, tmp_path
):
    TaskRun(tiny_model, tokenizer, config(tmp_path), data).run(plan(SETTING.children, drop=True))
    full = read(items_path(tmp_path, "gsm8k", "group/k0,1"))
    for k in range(SETTING.children):
        for before, after in zip(
            full, read(items_path(tmp_path, "gsm8k", f"drop/k{k}")), strict=True
        ):
            half = len(before["tokens"]) // 2
            assert after["tokens"][:half] == before["tokens"][:half]


def test_a_finished_run_evaluates_nothing_again(tiny_model, tokenizer, data, tmp_path, monkeypatch):
    run = TaskRun(tiny_model, tokenizer, config(tmp_path), data)
    run.run(plan(SETTING.children, drop=True))
    calls = []
    monkeypatch.setattr(runner_module, "generate", lambda *a: calls.append(a))
    TaskRun(tiny_model, tokenizer, config(tmp_path), data).run(plan(SETTING.children, drop=True))
    assert calls == []


def test_a_drop_needs_the_full_group_first(tiny_model, tokenizer, data, tmp_path):
    drops = [c for c in plan(SETTING.children, drop=True) if c.kind == "drop"]
    with pytest.raises(FileNotFoundError, match="must run first"):
        TaskRun(tiny_model, tokenizer, config(tmp_path), data).run(drops)


def test_confirmation_runs_log_round_to_nearest_and_dataset_positions(
    tiny_model, tokenizer, data, tmp_path
):
    from dataclasses import replace

    shifted = replace(
        config(tmp_path), gsm8k=GSM8KConfig(offset=64, batch_size=2, max_new_tokens=4)
    )
    conditions = plan(SETTING.children, drop=False, full_only=True, rtn_bits=(6,))
    TaskRun(tiny_model, tokenizer, shifted, data).run(conditions)
    summary = {row["id"] for row in read(tmp_path / "results.jsonl")}
    assert "gsm8k/rtn/b6" in summary
    assert "gsm8k/group/k0,1" in summary
    assert [a["problem"] for a in read(items_path(tmp_path, "gsm8k", "rtn/b6"))] == [64, 65, 66]
