"""MMLU, 5-shot, scored by likelihood (``docs/preregistration-swarm.md``, secondary task).

Each prompt is the usual header for the question's subject, five worked questions from the
subject's dev split, then the question ending in "Answer:". The prediction is whichever of
" A", " B", " C" and " D" the model finds most likely next.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from datasets import load_dataset
from transformers import PreTrainedModel

MMLU = ("cais/mmlu", "all")
LETTERS = "ABCD"
SHOTS = 5


class Encoder(Protocol):
    """The parts of a tokenizer that scoring uses."""

    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Token ids of ``text``."""
        ...


@dataclass(frozen=True)
class Question:
    """One multiple-choice question.

    Attributes:
        subject: MMLU subject, such as "college_physics".
        question: The question text.
        choices: The four options, in order A to D.
        answer: Index of the correct option.
    """

    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int


def load_mmlu(split: str, limit: int | None = None) -> list[Question]:
    """Questions of an MMLU split, in dataset order; ``limit`` keeps the first few."""
    rows = load_dataset(*MMLU, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [Question(r["subject"], r["question"], tuple(r["choices"]), r["answer"]) for r in rows]


def shots_by_subject(questions: Sequence[Question]) -> dict[str, list[Question]]:
    """Group worked examples (the dev split) by subject."""
    grouped: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        grouped[question.subject].append(question)
    return dict(grouped)


def prompt(question: Question, shots: dict[str, list[Question]]) -> str:
    """The 5-shot prompt for ``question``, ending just before the answer letter.

    A question never serves as its own example, so smoke tests can run on the dev split.
    """
    subject = question.subject.replace("_", " ")
    examples = [q for q in shots.get(question.subject, []) if q != question][:SHOTS]
    header = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    worked = "".join(f"{_format(q)} {LETTERS[q.answer]}\n\n" for q in examples)
    return header + worked + _format(question)


def letter_ids(tokenizer: Encoder) -> list[int]:
    """Token ids of " A" to " D".

    Raises:
        ValueError: If a letter is not a single token.
    """
    ids = [tokenizer.encode(f" {letter}", add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"answer letters must be single tokens, got {ids}")
    return [i[0] for i in ids]


@torch.inference_mode()
def predict(
    model: PreTrainedModel, prompts: Sequence[Sequence[int]], letters: list[int], pad: int
) -> list[int]:
    """Index of the most likely answer letter after each prompt, scored as one batch."""
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(prompts):
        ids[i, width - len(row) :] = torch.tensor(row)
        mask[i, width - len(row) :] = 1
    device = model.get_output_embeddings().weight.device
    logits = model(
        input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=1
    ).logits
    return logits[:, -1, letters].argmax(dim=-1).tolist()


def _format(question: Question) -> str:
    options = "".join(f"\n{LETTERS[i]}. {c}" for i, c in enumerate(question.choices))
    return f"{question.question.strip()}{options}\nAnswer:"
