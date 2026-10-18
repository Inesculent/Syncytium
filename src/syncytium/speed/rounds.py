"""Synchronization rounds of a fused forward pass (``docs/preregistration-swarm.md``, H9).

Devices that each hold one child fuse by averaging the output of every quantized layer before
anything nonlinear reads it: the mean of the children's outputs is the output of the averaged
weights, and averaging must happen before a norm, activation or attention breaks linearity.
Layers that read the same input (q, k and v; gate and up) are averaged together in one round.

Rounds are traced from the architecture by running one token through a model on the meta
device, so counting them for a 9B model needs no weights.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM

from syncytium.models import quantized_weights


@dataclass(frozen=True)
class Round:
    """One averaging step of a fused forward pass.

    Attributes:
        layers: The quantized layers whose outputs are averaged together.
        values: Values each device contributes per token: the layers' summed output width.
        head: Whether these are the output logits, which greedy decoding only needs the
            argmax of.
    """

    layers: tuple[str, ...]
    values: int
    head: bool = False


def trace_rounds(model: nn.Module) -> list[Round]:
    """The rounds of one token's forward pass through ``model``, in order.

    Every linear and embedding layer whose weight is quantized is a fused layer; consecutive
    calls that read the very same input tensor share a round.
    """
    quantized = {id(w) for w in quantized_weights(model).values()}
    head = model.get_output_embeddings()
    calls: list[tuple[str, Tensor, int, bool]] = []

    def record(name: str, module: nn.Module):
        def hook(_: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
            calls.append((name, args[0], output.shape[-1], module is head))

        return hook

    handles = [
        module.register_forward_hook(record(name, module))
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding)) and id(module.weight) in quantized
    ]
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.zeros(1, 1, dtype=torch.long, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return _group(calls)


def model_rounds(name_or_path: str) -> list[Round]:
    """Rounds of a model named by a Hugging Face id or local path, built without weights."""
    config = AutoConfig.from_pretrained(name_or_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return trace_rounds(model)


def _group(calls: list[tuple[str, Tensor, int, bool]]) -> list[Round]:
    rounds: list[Round] = []
    previous: Tensor | None = None
    for name, inputs, width, is_head in calls:
        if rounds and inputs is previous:
            last = rounds[-1]
            rounds[-1] = Round((*last.layers, name), last.values + width, last.head or is_head)
        else:
            rounds.append(Round((name,), width, is_head))
        previous = inputs
    return rounds
