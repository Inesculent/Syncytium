"""Rebuilding the parent from children (``docs/method.md``, "Rebuild" and "Partial rebuild")."""

from collections.abc import Mapping, Sequence
from enum import StrEnum

import torch
from torch import Tensor

from syncytium.quant.grid import Grid
from syncytium.quant.stagger import child_weights, offset


class Estimator(StrEnum):
    """How weights are estimated from a strict subset of the children."""

    MIDPOINT = "midpoint"  # partial_weights: midpoint of the intersected intervals
    MEAN = "mean"  # mean_weights: average of the surviving children


def parent_codes(children: Sequence[Tensor]) -> Tensor:
    """Sum the codes of all N children: Q = Σ_k q_k = ⌊N x + φ⌋ by Hermite's identity.

    Raises:
        ValueError: If ``children`` is empty.
    """
    if not children:
        raise ValueError("at least one child is required")
    return torch.stack([codes.to(torch.int32) for codes in children]).sum(dim=0)


def parent_weights(codes: Tensor, grid: Grid, num_children: int, phase: float = 0.0) -> Tensor:
    """Dequantize the parent on the fine grid: ŵ = m + (Δ/N)(Q + ½ - φ)."""
    shift = 0.5 - phase
    return grid.minimum + grid.step / num_children * (codes.to(grid.step.dtype) + shift)


def partial_weights(
    available: Mapping[int, Tensor], grid: Grid, num_children: int, phase: float = 0.0
) -> Tensor:
    """Estimate the weights from any subset of children.

    Child k places x in [q_k - o_k, q_k + 1 - o_k), where o_k = (k + φ)/N. The estimate is
    the midpoint of the intersection of these intervals; with every child present it equals
    the parent.

    Args:
        available: Codes of the surviving children, keyed by child index k.
        grid: The shared grid.
        num_children: Number of children N in the original set.
        phase: Offset phase φ.

    Raises:
        ValueError: If ``available`` is empty.
    """
    if not available:
        raise ValueError("at least one child is required")
    starts = torch.stack(
        [
            codes.to(grid.step.dtype) - offset(index, num_children, phase)
            for index, codes in available.items()
        ]
    )
    midpoint = (starts.amax(dim=0) + starts.amin(dim=0) + 1) / 2
    return grid.minimum + grid.step * midpoint


def mean_weights(
    available: Mapping[int, Tensor], grid: Grid, num_children: int, phase: float = 0.0
) -> Tensor:
    """Average the dequantized weights of any subset of children.

    The average is linear in the children, so devices that each hold one child can apply it
    by averaging their layer outputs, without ever assembling the weights. With every child
    present it equals the parent exactly; with one child it is that child. With some
    children missing it is slightly less accurate than :func:`partial_weights`.

    Args:
        available: Codes of the surviving children, keyed by child index k.
        grid: The shared grid.
        num_children: Number of children N in the original set.
        phase: Offset phase φ.

    Raises:
        ValueError: If ``available`` is empty.
    """
    if not available:
        raise ValueError("at least one child is required")
    weights = [
        child_weights(codes, grid, index, num_children, phase) for index, codes in available.items()
    ]
    return torch.stack(weights).mean(dim=0)
