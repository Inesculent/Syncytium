"""Simulated quantization of whole weight tensors.

Each function quantizes a tensor group by group and returns dequantized weights in the
tensor's original shape. Phase 1 evaluates children and rebuilds this way, without custom
low-bit kernels.
"""

from collections.abc import Callable, Collection

import torch
from torch import Tensor

from holograph.quant.grid import fit_grid, from_groups, normalize, to_groups
from holograph.quant.rebuild import parent_codes, parent_weights, partial_weights
from holograph.quant.scheme import Scheme
from holograph.quant.stagger import child_codes, child_weights

# Groups processed at once. Bounds temporary memory for large tensors such as embeddings.
CHUNK_GROUPS = 1 << 18


def simulate_child(weight: Tensor, scheme: Scheme, index: int) -> Tensor:
    """Dequantized weights of child ``index`` (k) under ``scheme``."""

    def quantize(groups: Tensor) -> Tensor:
        grid = fit_grid(groups, scheme.bits)
        codes = child_codes(normalize(groups, grid), grid, index, scheme.num_children)
        return child_weights(codes, grid, index, scheme.num_children)

    return _map_groups(weight, scheme.group_size, quantize)


def simulate_rebuild(weight: Tensor, scheme: Scheme, indices: Collection[int]) -> Tensor:
    """Weights rebuilt from the children in ``indices``.

    With every child present this is the exact integer rebuild; with a subset it is the
    partial rebuild by interval intersection.
    """
    available = sorted(set(indices))

    def rebuild(groups: Tensor) -> Tensor:
        grid = fit_grid(groups, scheme.bits)
        x = normalize(groups, grid)
        codes = {k: child_codes(x, grid, k, scheme.num_children) for k in available}
        if len(codes) == scheme.num_children:
            return parent_weights(parent_codes(list(codes.values())), grid, scheme.num_children)
        return partial_weights(codes, grid, scheme.num_children)

    return _map_groups(weight, scheme.group_size, rebuild)


def _map_groups(weight: Tensor, group_size: int, fn: Callable[[Tensor], Tensor]) -> Tensor:
    """Apply ``fn`` to the groups of ``weight`` in chunks and restore its shape."""
    groups = to_groups(weight, group_size)
    result = torch.cat([fn(chunk) for chunk in groups.split(CHUNK_GROUPS)])
    return from_groups(result, weight.shape)
