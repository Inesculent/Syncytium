"""Batched greedy generation that records each answer's token entropy.

Prompts are left-padded so a batch decodes together; padding changes no output (checked in
``tests/tasks/test_generation.py``). Each row stops at its first stop sequence, and the
entropy of every predictive distribution is recorded for H8's escalation signal.

:func:`continue_after_switch` serves H13: one set of weights builds the cache for a prompt and
a forced prefix of the answer, then ``switch`` swaps in other weights, which continue from that
cache.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor
from transformers import LogitsProcessor, PreTrainedModel, StoppingCriteria


class Tokenizer(Protocol):
    """The parts of a tokenizer that generation uses."""

    pad_token_id: int | None
    eos_token_id: int | None

    def decode(self, ids: Sequence[int]) -> str:
        """Text of ``ids``."""
        ...


@dataclass(frozen=True)
class Decoding:
    """How answers are generated.

    Attributes:
        max_new_tokens: Most answer tokens per prompt.
        stops: An answer ends at the first of these; it is not part of the answer.
    """

    max_new_tokens: int
    stops: tuple[str, ...]


@dataclass(frozen=True)
class Completion:
    """One generated answer.

    Attributes:
        tokens: Answer tokens up to the first stop sequence, which is not included.
        text: The answer text, cut at the first stop sequence.
        entropy: Mean entropy (nats) of the predictive distributions of the kept tokens.
    """

    tokens: list[int]
    text: str
    entropy: float


class _EntropyRecorder(LogitsProcessor):
    """Records the entropy of every row's next-token distribution, leaving scores unchanged."""

    def __init__(self) -> None:
        self.steps: list[Tensor] = []

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        logp = scores.float().log_softmax(dim=-1)
        self.steps.append(-(logp.exp() * logp).sum(dim=-1).cpu())
        return scores


class _StopOnText(StoppingCriteria):
    """Stops each row once its answer text, from position ``starts[row]``, has a stop."""

    def __init__(self, tokenizer: Tokenizer, starts: Sequence[int], stops: Sequence[str]) -> None:
        self.tokenizer, self.starts, self.stops = tokenizer, starts, stops

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs: Any) -> Tensor:
        done = [
            any(stop in self.tokenizer.decode(row[start:].tolist()) for stop in self.stops)
            for row, start in zip(input_ids, self.starts, strict=True)
        ]
        return torch.tensor(done, device=input_ids.device)


def generate(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    prompts: Sequence[Sequence[int]],
    decoding: Decoding,
) -> list[Completion]:
    """Greedy answers to ``prompts`` (token ids), decoded together as one batch."""
    ids, mask = _left_pad(prompts, _pad_id(tokenizer), _device(model))
    recorder = _EntropyRecorder()
    stop = _StopOnText(tokenizer, [ids.shape[1]] * len(prompts), decoding.stops)
    out = _greedy(
        model,
        tokenizer,
        ids,
        mask,
        decoding.max_new_tokens,
        logits_processor=[recorder],
        stopping_criteria=[stop],
    )
    entropies = torch.stack(recorder.steps, dim=1)
    return [
        _complete(tokenizer, row[ids.shape[1] :].tolist(), decoding.stops, entropies[i])
        for i, row in enumerate(out.sequences)
    ]


def continue_after_switch(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    prompts: Sequence[Sequence[int]],
    prefixes: Sequence[Sequence[int]],
    switch: Callable[[], None],
    decoding: Decoding,
) -> list[Completion]:
    """Answers whose start comes from the current weights and whose rest from switched ones.

    The current weights read each prompt and its forced prefix and choose one more token;
    the cache they built is kept. ``switch`` then replaces the weights, and the new weights
    generate the rest of each answer from that cache, up to ``decoding.max_new_tokens``
    answer tokens in all. Entropies are not recorded.
    """
    joined = [[*p, *f] for p, f in zip(prompts, prefixes, strict=True)]
    ids, mask = _left_pad(joined, _pad_id(tokenizer), _device(model))
    first = _greedy(model, tokenizer, ids, mask, 1)
    switch()
    sequences = first.sequences
    mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=1)
    starts = [ids.shape[1] - len(f) for f in prefixes]
    # The row with the shortest prefix has the most answer left to write.
    budget = decoding.max_new_tokens - 1 - min(map(len, prefixes))
    if budget > 0:
        stop = _StopOnText(tokenizer, starts, decoding.stops)
        sequences = _greedy(
            model,
            tokenizer,
            sequences,
            mask,
            budget,
            stopping_criteria=[stop],
            past_key_values=first.past_key_values,
        ).sequences
    unrecorded = torch.full((decoding.max_new_tokens,), float("nan"))
    return [
        _complete(
            tokenizer, row[s:].tolist()[: decoding.max_new_tokens], decoding.stops, unrecorded
        )
        for row, s in zip(sequences, starts, strict=True)
    ]


def _greedy(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    ids: Tensor,
    mask: Tensor,
    max_new_tokens: int,
    **options: Any,
) -> Any:
    """``model.generate`` with greedy decoding; ``options`` are passed on unchanged."""
    return model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=_pad_id(tokenizer),
        return_dict_in_generate=True,
        **options,
    )


def _complete(
    tokenizer: Tokenizer, tokens: list[int], stops: Sequence[str], entropies: Tensor
) -> Completion:
    """Cut generated ``tokens`` at the end of sequence and at the first stop sequence."""
    if tokenizer.eos_token_id in tokens:
        tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
    text = tokenizer.decode(tokens)
    cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=len(text))
    # Generation stops right after a stop appears, so only a token or two is dropped here.
    kept = len(tokens)
    while kept and len(tokenizer.decode(tokens[:kept])) > cut:
        kept -= 1
    return Completion(tokens[:kept], text[:cut], entropies[: max(kept, 1)].mean().item())


def _left_pad(
    rows: Sequence[Sequence[int]], pad: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Left-padded ids and their attention mask."""
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            ids[i, width - len(row) :] = torch.tensor(row)
            mask[i, width - len(row) :] = 1
    return ids.to(device), mask.to(device)


def _pad_id(tokenizer: Tokenizer) -> int:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("the tokenizer has neither a pad nor an end-of-sequence token")
    return pad


def _device(model: PreTrainedModel) -> torch.device:
    return model.get_output_embeddings().weight.device
