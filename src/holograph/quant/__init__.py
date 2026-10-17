"""Staggered quantization: functional low-bit children and exact integer rebuild."""

from holograph.quant.grid import Grid, fit_grid, from_groups, normalize, to_groups
from holograph.quant.rebuild import parent_codes, parent_weights, partial_weights
from holograph.quant.scheme import Scheme
from holograph.quant.simulate import simulate_child, simulate_rebuild
from holograph.quant.stagger import child_codes, child_weights, offset, quantize_children

__all__ = [
    "Grid",
    "Scheme",
    "child_codes",
    "child_weights",
    "fit_grid",
    "from_groups",
    "normalize",
    "offset",
    "parent_codes",
    "parent_weights",
    "partial_weights",
    "quantize_children",
    "simulate_child",
    "simulate_rebuild",
    "to_groups",
]
