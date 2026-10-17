"""Placing weights on the stages of a pooled pipeline (``docs/method.md``, "Pooled pipeline").

Stage 0 holds every weight outside the decoder layers: the embedding (which Qwen3.5 ties to
the output head) and the final norm. Stages 1 … S hold contiguous blocks of decoder layers
of near-equal size. Each stage is served by N devices, one per child.
"""

import re
from collections.abc import Iterable

from torch import nn

from syncytium.eval import Footprint, ParameterCounts, child_footprint
from syncytium.models import split_parameters
from syncytium.quant import Scheme

LAYER = re.compile(r"\.layers\.(\d+)\.")


def stage_of(name: str, num_layers: int, num_stages: int) -> int:
    """Stage of parameter ``name``: 0 outside the decoder layers, else 1 … ``num_stages``."""
    match = LAYER.search(name)
    if match is None:
        return 0
    return 1 + int(match.group(1)) * num_stages // num_layers


def assign_stages(names: Iterable[str], num_stages: int) -> dict[str, int]:
    """Map each parameter name to its stage.

    Raises:
        ValueError: If there are fewer decoder layers than stages.
    """
    names = list(names)
    layers = {int(match.group(1)) for name in names if (match := LAYER.search(name))}
    if len(layers) < num_stages:
        raise ValueError(f"{len(layers)} decoder layers cannot fill {num_stages} stages")
    return {name: stage_of(name, len(layers), num_stages) for name in names}


def device_footprints(model: nn.Module, scheme: Scheme, num_stages: int) -> list[Footprint]:
    """Weight memory of one device on each stage: one child of that stage's weights."""
    quantized, full = split_parameters(model)
    stages = assign_stages([*quantized, *full], num_stages)

    def count(params: dict[str, nn.Parameter], stage: int) -> int:
        return sum(p.numel() for name, p in params.items() if stages[name] == stage)

    return [
        child_footprint(ParameterCounts(count(quantized, stage), count(full, stage)), scheme)
        for stage in range(num_stages + 1)
    ]
