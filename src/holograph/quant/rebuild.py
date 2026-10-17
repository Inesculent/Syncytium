"""Rebuilding the parent from children (``docs/method.md``, "Rebuild" and "Partial rebuild")."""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from holograph.quant.grid import Grid
from holograph.quant.stagger import offset


def parent_codes(children: Sequence[Tensor]) -> Tensor:
    """Sum the codes of all N children: Q = Σ_k q_k = ⌊N x⌋ by Hermite's identity.

    Raises:
        ValueError: If ``children`` is empty.
    """
    if not children:
        raise ValueError("at least one child is required")
    return torch.stack([codes.to(torch.int32) for codes in children]).sum(dim=0)


def parent_weights(codes: Tensor, grid: Grid, num_children: int) -> Tensor:
    """Dequantize the parent on the fine grid: ŵ = m + (Δ/N)(Q + ½)."""
    return grid.minimum + grid.step / num_children * (codes.to(grid.step.dtype) + 0.5)


def partial_weights(available: Mapping[int, Tensor], grid: Grid, num_children: int) -> Tensor:
    """Estimate the weights from any subset of children.

    Child k places x in [q_k - k/N, q_k + 1 - k/N). The estimate is the midpoint of the
    intersection of these intervals; with every child present it equals the parent.

    Args:
        available: Codes of the surviving children, keyed by child index k.
        grid: The shared grid.
        num_children: Number of children N in the original set.

    Raises:
        ValueError: If ``available`` is empty.
    """
    if not available:
        raise ValueError("at least one child is required")
    starts = torch.stack(
        [
            codes.to(grid.step.dtype) - offset(index, num_children)
            for index, codes in available.items()
        ]
    )
    midpoint = (starts.amax(dim=0) + starts.amin(dim=0) + 1) / 2
    return grid.minimum + grid.step * midpoint
