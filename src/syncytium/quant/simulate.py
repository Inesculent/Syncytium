"""Simulated quantization of whole weight tensors.

Each function quantizes a tensor group by group and returns dequantized weights in the
tensor's original shape. Phase 1 evaluates children and rebuilds this way, without custom
low-bit kernels. With a rotation seed, groups are rotated before quantization and rotated
back afterwards.
"""

from collections.abc import Callable, Collection

import torch
from torch import Tensor

from syncytium.quant.grid import fit_grid, from_groups, normalize, to_groups
from syncytium.quant.rebuild import (
    Estimator,
    mean_weights,
    parent_codes,
    parent_weights,
    partial_weights,
)
from syncytium.quant.rotation import random_rotation
from syncytium.quant.scheme import Scheme
from syncytium.quant.stagger import child_codes, child_weights

# Groups processed at once. Bounds temporary memory for large tensors such as embeddings.
CHUNK_GROUPS = 1 << 18


def simulate_child(weight: Tensor, scheme: Scheme, index: int) -> Tensor:
    """Dequantized weights of child ``index`` (k) under ``scheme``."""
    n, phase = scheme.num_children, scheme.phase

    def quantize(groups: Tensor) -> Tensor:
        grid = fit_grid(groups, scheme.bits)
        codes = child_codes(normalize(groups, grid), grid, index, n, phase)
        return child_weights(codes, grid, index, n, phase)

    return _map_groups(weight, scheme, quantize)


def simulate_rebuild(
    weight: Tensor,
    scheme: Scheme,
    indices: Collection[int],
    estimator: Estimator = Estimator.MIDPOINT,
) -> Tensor:
    """Weights rebuilt from the children in ``indices``.

    With every child present this is the exact integer rebuild; with a subset, ``estimator``
    chooses between the partial rebuild by interval intersection and the children's mean.
    """
    available = sorted(set(indices))
    n, phase = scheme.num_children, scheme.phase
    partial = mean_weights if estimator == Estimator.MEAN else partial_weights

    def rebuild(groups: Tensor) -> Tensor:
        grid = fit_grid(groups, scheme.bits)
        x = normalize(groups, grid)
        codes = {k: child_codes(x, grid, k, n, phase) for k in available}
        if len(codes) == n:
            return parent_weights(parent_codes(list(codes.values())), grid, n, phase)
        return partial(codes, grid, n, phase)

    return _map_groups(weight, scheme, rebuild)


def _map_groups(weight: Tensor, scheme: Scheme, fn: Callable[[Tensor], Tensor]) -> Tensor:
    """Apply ``fn`` to the (rotated) groups of ``weight`` in chunks and restore its shape."""
    groups = to_groups(weight, scheme.group_size)
    if scheme.rotation_seed is not None:
        rotation = random_rotation(scheme.group_size, scheme.rotation_seed).to(weight)
        fn = _in_rotated_basis(fn, rotation)
    result = torch.cat([fn(chunk) for chunk in groups.split(CHUNK_GROUPS)])
    return from_groups(result, weight.shape)


def _in_rotated_basis(
    fn: Callable[[Tensor], Tensor], rotation: Tensor
) -> Callable[[Tensor], Tensor]:
    """Wrap ``fn`` so it runs on groups times R and its output is multiplied back by Rᵀ."""
    return lambda groups: fn(groups @ rotation) @ rotation.T
