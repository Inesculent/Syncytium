"""Per-group asymmetric quantization grids shared by every child.

Notation follows ``docs/method.md``: a group of weights with minimum m and maximum M is
quantized with step Δ = (M - m) / (2^b - 1).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

MAX_BITS = 8


@dataclass(frozen=True)
class Grid:
    """Quantization grid for a grouped weight tensor.

    Attributes:
        minimum: Per-group minimum m, shape ``(groups, 1)``.
        step: Per-group step Δ, shape ``(groups, 1)``.
        bits: Bits per child code, b.
    """

    minimum: Tensor
    step: Tensor
    bits: int

    @property
    def levels(self) -> int:
        """Number of distinct child codes, 2^b."""
        return 1 << self.bits


def to_groups(weight: Tensor, group_size: int) -> Tensor:
    """Reshape a weight tensor into rows of ``group_size`` consecutive values.

    Groups run along the last (input) dimension.

    Raises:
        ValueError: If the last dimension is not divisible by ``group_size``.
    """
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {weight.shape[-1]} is not divisible by group size {group_size}"
        )
    return weight.reshape(-1, group_size)


def from_groups(groups: Tensor, shape: torch.Size) -> Tensor:
    """Restore the original shape of a tensor produced by :func:`to_groups`."""
    return groups.reshape(shape)


def fit_grid(groups: Tensor, bits: int, storage: torch.dtype | None = None) -> Grid:
    """Fit the grid of each group: m = min(w) and Δ = (M - m) / (2^b - 1).

    A constant group gets the smallest positive step, so its weights map to code 0 and
    dequantize back to m up to a negligible error.

    Args:
        groups: Weights arranged in groups, from :func:`to_groups`.
        bits: Bits per child code, b.
        storage: If given, m and Δ are made exactly representable in this dtype (as in
            shard files), rounding m down and Δ up so every code still fits in b bits.

    Raises:
        ValueError: If ``bits`` is outside ``[1, 8]``.
    """
    if not 1 <= bits <= MAX_BITS:
        raise ValueError(f"bits must be in [1, {MAX_BITS}], got {bits}")
    minimum = groups.amin(dim=1, keepdim=True)
    if storage is not None:
        minimum = _round_toward(minimum, storage, float("-inf"))
    span = groups.amax(dim=1, keepdim=True) - minimum
    smallest = torch.finfo(storage or groups.dtype).tiny
    step = (span / ((1 << bits) - 1)).clamp_min(smallest)
    if storage is not None:
        step = _round_toward(step, storage, float("inf"))
    return Grid(minimum=minimum, step=step, bits=bits)


def _round_toward(values: Tensor, dtype: torch.dtype, direction: float) -> Tensor:
    """Nearest value representable in ``dtype`` on the ``direction`` side of ``values``."""
    rounded = values.to(dtype)
    back = rounded.to(values.dtype)
    wrong_side = back > values if direction < 0 else back < values
    stepped = torch.nextafter(rounded, torch.full_like(rounded, direction))
    return torch.where(wrong_side, stepped, rounded).to(values.dtype)


def normalize(groups: Tensor, grid: Grid) -> Tensor:
    """Express weights in grid units: x = (w - m) / Δ, which lies in [0, 2^b - 1]."""
    return (groups - grid.minimum) / grid.step
