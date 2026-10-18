"""How often a lone agent's drafts survive the fused group's check (H9).

With greedy decoding, the fused group accepts a drafted token when it is the token the group
would have chosen itself. On fixed text, each position therefore either agrees (the lone
child's next-token argmax equals the full group's) or not. Speculative decoding is simulated
along that text: from each position, the drafts run until the first disagreement, at most k,
and the group adds one token of its own. No independence between positions is assumed.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import PreTrainedModel

# Positions passed through the output head at once, as in perplexity evaluation.
HEAD_CHUNK = 512


@torch.inference_mode()
def greedy_choices(model: PreTrainedModel, batches: Tensor) -> Tensor:
    """The argmax next token at every position of every row of ``batches``.

    Returns:
        ``int32`` tensor of shape ``(rows, length)``, on the CPU.
    """
    device = model.get_output_embeddings().weight.device
    head = model.get_output_embeddings()
    rows = []
    for row in batches:
        hidden = model.get_decoder()(row.unsqueeze(0).to(device)).last_hidden_state[0]
        rows.append(
            torch.cat([head(part).argmax(dim=-1).int().cpu() for part in hidden.split(HEAD_CHUNK)])
        )
    return torch.stack(rows)


def tokens_per_cycle(agree: Tensor, draft: int) -> float:
    """Mean tokens produced per draft-and-check cycle, simulated along ``agree``.

    Args:
        agree: Boolean ``(rows, positions)``: whether the drafter's choice matches the group's.
        draft: Tokens drafted per cycle, k ≥ 1.
    """
    if draft < 1:
        raise ValueError(f"draft must be at least 1, got {draft}")
    produced, cycles = 0, 0
    for row in agree.tolist():
        position = 0
        while position < len(row):
            run = 0
            while run < draft and position + run < len(row) and row[position + run]:
                run += 1
            step = min(run + 1, len(row) - position)
            produced, cycles = produced + step, cycles + 1
            position += step
    return produced / cycles


def agreement_curve(agree: Tensor, drafts: Sequence[int]) -> dict[int, float]:
    """:func:`tokens_per_cycle` for each draft length in ``drafts``."""
    return {k: tokens_per_cycle(agree, k) for k in drafts}
