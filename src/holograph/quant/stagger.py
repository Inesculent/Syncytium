"""Staggered child quantization (``docs/method.md``, "Staggered quantization").

Child k of N rounds on the shared grid shifted by k/N steps: q_k = ⌊x + k/N⌋.
"""

import torch
from torch import Tensor

from holograph.quant.grid import Grid, normalize


def offset(index: int, num_children: int) -> float:
    """Grid offset k/N of child ``index``, in units of Δ.

    Raises:
        ValueError: If ``index`` is not in ``[0, num_children)``.
    """
    if not 0 <= index < num_children:
        raise ValueError(f"child index {index} is not in [0, {num_children})")
    return index / num_children


def child_codes(x: Tensor, grid: Grid, index: int, num_children: int) -> Tensor:
    """Codes of one child, q_k = ⌊x + k/N⌋, as ``uint8``.

    Args:
        x: Weights in grid units, from :func:`~holograph.quant.grid.normalize`.
        grid: The shared grid.
        index: Child index k.
        num_children: Number of children N.
    """
    codes = torch.floor(x + offset(index, num_children))
    # Codes lie in [0, 2^b - 1] exactly; the clamp only absorbs float rounding at the edges.
    return codes.clamp_(0, grid.levels - 1).to(torch.uint8)


def child_weights(codes: Tensor, grid: Grid, index: int, num_children: int) -> Tensor:
    """Dequantize one child: ŵ_k = m + Δ (q_k + ½ - k/N)."""
    shift = 0.5 - offset(index, num_children)
    return grid.minimum + grid.step * (codes.to(grid.step.dtype) + shift)


def quantize_children(groups: Tensor, grid: Grid, num_children: int) -> list[Tensor]:
    """Codes of all N children, ordered by child index."""
    x = normalize(groups, grid)
    return [child_codes(x, grid, index, num_children) for index in range(num_children)]
