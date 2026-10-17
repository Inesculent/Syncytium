"""Shared randomized Hadamard rotation (``docs/method.md``, "Optional shared rotation").

Each group of g weights is multiplied by the same orthogonal g-by-g matrix R = D H / √g, where
H is a Sylvester Hadamard matrix and D a diagonal of random signs fixed by a seed. Rotation
spreads outliers across the group before rounding; because R is linear and shared by every
child, the rebuild identity still holds after rotating back.
"""

import torch
from torch import Tensor


def hadamard(size: int) -> Tensor:
    """Sylvester Hadamard matrix of order ``size``, with entries ±1.

    Raises:
        ValueError: If ``size`` is not a power of two.
    """
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)])
    return matrix


def random_rotation(size: int, seed: int) -> Tensor:
    """Orthogonal matrix R = D H / √size with random signs D drawn from ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (size, 1), generator=generator) * 2 - 1
    return signs * hadamard(size) / size**0.5
