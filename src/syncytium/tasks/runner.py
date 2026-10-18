"""Running the swarm task evaluation (``docs/preregistration-swarm.md``, Gate A).

Each condition writes one summary row per task to ``results.jsonl`` and every answer to
``items/<task>/<condition>.jsonl``, which the paired statistics of H7, H8, H11 and H13 need.
Rows are written as each condition finishes, so an interrupted run resumes where it stopped.
"""

import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel

from syncytium.eval import load_wikitext2, windows
from syncytium.models import WeightBank, load_model, load_tokenizer, quantized_weights
from syncytium.quant import ChannelScales, Estimator
from syncytium.sweep.records import ResultLog, append_run_metadata
from syncytium.sweep.runner import calibrate
from syncytium.tasks import gsm8k, mmlu
from syncytium.tasks.conditions import Kind, TaskCondition, plan
from syncytium.tasks.config import TaskConfig
from syncytium.tasks.generation import Completion, Decoding, continue_after_switch, generate

logger = logging.getLogger(__name__)

GSM8K, MMLU = "gsm8k", "mmlu"


@dataclass(frozen=True)
class TaskData:
    """Tokenized prompts and references for each task in a run (empty when not run)."""

    problems: Sequence[gsm8k.Problem] = ()
    problem_ids: Sequence[list[int]] = ()
    questions: Sequence[mmlu.Question] = ()
    question_ids: Sequence[list[int]] = ()


def run_tasks(config: TaskConfig, config_path: Path) -> None:
    """Load the model and tasks named in ``config`` and evaluate every condition."""
    append_run_metadata(config.output / "runs.jsonl", config_path)
    model = load_model(config.model.path, config.model.device, getattr(torch, config.model.dtype))
    tokenizer = load_tokenizer(config.model.path)
    scales = None
    if config.setting.scaling:
        evaluation = config.evaluation
        train = load_wikitext2(tokenizer, split="train")
        scales = calibrate(
            model, windows(train, evaluation.calibration_window, evaluation.calibration_windows)
        )
    run = TaskRun(model, tokenizer, config, load_data(config, tokenizer), scales)
    run.run(plan(config.setting.children, config.setting.drop))


def load_data(config: TaskConfig, tokenizer: Any) -> TaskData:
    """Load and tokenize the tasks a run evaluates."""
    problems, questions, shots = [], [], {}
    if config.gsm8k is not None:
        problems = gsm8k.load_gsm8k(config.gsm8k.split, config.gsm8k.limit)
    if config.mmlu is not None:
        questions = mmlu.load_mmlu(config.mmlu.split, config.mmlu.limit)
        shots = mmlu.shots_by_subject(mmlu.load_mmlu("dev"))
    return TaskData(
        problems,
        [_encode(tokenizer, gsm8k.prompt(p.question)) for p in problems],
        questions,
        [_encode(tokenizer, mmlu.prompt(q, shots)) for q in questions],
    )


class TaskRun:
    """Evaluates conditions on the tasks of one run.

    Args:
        model: The loaded bf16 model; its quantized weights are swapped per condition.
        tokenizer: The model's tokenizer.
        config: The run's configuration.
        data: Tokenized prompts, from :func:`load_data`.
        scales: Channel scales, needed when the setting scales its channels.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        config: TaskConfig,
        data: TaskData,
        scales: ChannelScales | None = None,
    ) -> None:
        self.model, self.tokenizer, self.config, self.data = model, tokenizer, config, data
        self.scales = scales
        self.bank = WeightBank(quantized_weights(model))
        self.log = ResultLog(config.output / "results.jsonl")
        self._full: dict[str, Tensor] | None = None

    def run(self, conditions: Sequence[TaskCondition]) -> None:
        """Evaluate every task of every condition not already logged."""
        done = self.log.completed()
        for position, condition in enumerate(conditions, start=1):
            tasks = [t for t in self._tasks(condition) if f"{t}/{condition.id}" not in done]
            if not tasks:
                continue
            if condition.kind == Kind.DROP:
                self._run_drop(condition)
                continue
            self._apply(condition)
            for task in tasks:
                start = time.perf_counter()
                items = self._gsm8k() if task == GSM8K else self._mmlu()
                self._record(task, condition, items, time.perf_counter() - start)
            logger.info("[%d/%d] %s done", position, len(conditions), condition.id)

    def _tasks(self, condition: TaskCondition) -> list[str]:
        tasks = [GSM8K] if self.data.problems else []
        if self.data.questions and condition.kind != Kind.DROP:
            tasks.append(MMLU)
        return tasks

    def _apply(self, condition: TaskCondition) -> None:
        estimator = Estimator(self.config.setting.estimator)
        transform = condition.transform(self.config.setting.scheme, estimator, self.scales)
        if transform is None:
            self.bank.restore()
        else:
            self.bank.apply_named(transform)

    def _gsm8k(self) -> list[dict[str, Any]]:
        settings = self.config.gsm8k
        decoding = Decoding(settings.max_new_tokens, gsm8k.STOPS)
        completions: dict[int, Completion] = {}
        for batch in _batches(self.data.problem_ids, settings.batch_size):
            prompts = [self.data.problem_ids[i] for i in batch]
            answers = generate(self.model, self.tokenizer, prompts, decoding)
            completions.update(zip(batch, answers, strict=True))
        return [self._gsm8k_item(i, completions[i]) for i in range(len(completions))]

    def _run_drop(self, condition: TaskCondition) -> None:
        """H13: the full group writes the first half of each answer, the survivors the rest."""
        start = time.perf_counter()
        full_group = TaskCondition(Kind.GROUP, condition.indices)
        reference = self._read_items(GSM8K, full_group)
        prefixes = [row["tokens"][: len(row["tokens"]) // 2] for row in reference]
        if self._full is None:
            self._apply(full_group)
            self._full = self.bank.snapshot()
        self._apply(condition.survivors)
        survivors = self.bank.snapshot()
        settings = self.config.gsm8k
        decoding = Decoding(settings.max_new_tokens, gsm8k.STOPS)
        completions: dict[int, Completion] = {}
        for batch in _batches(self.data.problem_ids, settings.batch_size):
            self.bank.load(self._full)
            answers = continue_after_switch(
                self.model,
                self.tokenizer,
                [self.data.problem_ids[i] for i in batch],
                [prefixes[i] for i in batch],
                lambda: self.bank.load(survivors),
                decoding,
            )
            completions.update(zip(batch, answers, strict=True))
        items = [self._gsm8k_item(i, completions[i]) for i in range(len(completions))]
        self._record(GSM8K, condition, items, time.perf_counter() - start)
        logger.info("%s done", condition.id)

    def _gsm8k_item(self, index: int, completion: Completion) -> dict[str, Any]:
        gold = self.data.problems[index].answer
        prediction = gsm8k.extract(completion.text)
        return {
            "index": index,
            "gold": gold,
            "prediction": prediction,
            "correct": gsm8k.is_correct(prediction, gold),
            "entropy": completion.entropy,
            "tokens": completion.tokens,
            "text": completion.text,
        }

    def _mmlu(self) -> list[dict[str, Any]]:
        letters = mmlu.letter_ids(self.tokenizer)
        pad = self.tokenizer.pad_token_id
        predictions: dict[int, int] = {}
        for batch in _batches(self.data.question_ids, self.config.mmlu.batch_size):
            prompts = [self.data.question_ids[i] for i in batch]
            predictions.update(
                zip(batch, mmlu.predict(self.model, prompts, letters, pad), strict=True)
            )
        return [
            {
                "index": i,
                "subject": q.subject,
                "gold": q.answer,
                "prediction": predictions[i],
                "correct": predictions[i] == q.answer,
            }
            for i, q in enumerate(self.data.questions)
        ]

    def _record(
        self, task: str, condition: TaskCondition, items: list[dict[str, Any]], seconds: float
    ) -> None:
        path = items_path(self.config.output, task, condition.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(item) + "\n" for item in items)
        correct = sum(item["correct"] for item in items)
        self.log.append(
            {
                "id": f"{task}/{condition.id}",
                "task": task,
                "condition": condition.id,
                "kind": condition.kind.value,
                "indices": list(condition.indices),
                "dropped": condition.dropped,
                "estimator": self.config.setting.estimator,
                "accuracy": correct / len(items),
                "correct": correct,
                "total": len(items),
                "seconds": round(seconds, 2),
            }
        )
        logger.info("%s %s: %d/%d correct", task, condition.id, correct, len(items))

    def _read_items(self, task: str, condition: TaskCondition) -> list[dict[str, Any]]:
        path = items_path(self.config.output, task, condition.id)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; {condition.id} must run first")
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]


def items_path(output: Path, task: str, condition_id: str) -> Path:
    """Where the per-answer rows of one task and condition are written."""
    return output / "items" / task / f"{condition_id.replace('/', '-')}.jsonl"


def _batches(prompts: Sequence[Sequence[int]], size: int) -> Iterator[list[int]]:
    """Indices of ``prompts`` in batches of similar length, longest first.

    Longest first, so a batch that does not fit in memory fails at the start of a run.
    """
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for start in range(0, len(order), size):
        yield order[start : start + size]


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)
