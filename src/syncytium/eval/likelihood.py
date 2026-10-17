"""Perplexity over fixed, non-overlapping windows."""

import math

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from transformers import PreTrainedModel

# Positions passed through the output head at once. Full-vocabulary logits for a whole
# window would take gigabytes in float32.
HEAD_CHUNK = 512


def windows(ids: Tensor, length: int, limit: int | None = None) -> Tensor:
    """Split 1-D token ids into non-overlapping windows of ``length``, dropping the remainder.

    Args:
        ids: Token ids of the whole corpus.
        length: Tokens per window.
        limit: Keep only the first ``limit`` windows, for quick runs.

    Raises:
        ValueError: If ``ids`` is shorter than one window.
    """
    count = ids.numel() // length
    if count == 0:
        raise ValueError(f"{ids.numel()} tokens is shorter than one window of {length}")
    if limit is not None:
        count = min(count, limit)
    return ids[: count * length].reshape(count, length)


@torch.inference_mode()
def perplexity(model: PreTrainedModel, batches: Tensor) -> float:
    """Perplexity of ``model`` over every predicted token in ``batches``.

    Each row of ``batches`` is evaluated independently; the first token of a row has no
    prediction. Log-likelihoods are accumulated in float32.
    """
    device = model.get_output_embeddings().weight.device
    total_nll, total_tokens = 0.0, 0
    for row in batches:
        ids = row.unsqueeze(0).to(device)
        total_nll += _window_nll(model, ids)
        total_tokens += ids.shape[1] - 1
    return math.exp(total_nll / total_tokens)


def _window_nll(model: PreTrainedModel, ids: Tensor) -> float:
    """Summed negative log-likelihood of one window, computing logits in chunks."""
    hidden = model.get_decoder()(ids).last_hidden_state[:, :-1]
    targets = ids[:, 1:]
    head = model.get_output_embeddings()
    nll = 0.0
    for states, labels in zip(
        hidden.split(HEAD_CHUNK, dim=1), targets.split(HEAD_CHUNK, dim=1), strict=True
    ):
        logits = head(states).float()
        nll += cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum").item()
    return nll
