"""Staggered quantization: functional low-bit children and exact integer rebuild."""

from syncytium.quant.grid import Grid, fit_grid, from_groups, normalize, to_groups
from syncytium.quant.rebuild import (
    Estimator,
    mean_weights,
    parent_codes,
    parent_weights,
    partial_weights,
)
from syncytium.quant.scaling import ChannelScales, search_scales
from syncytium.quant.scheme import Scheme
from syncytium.quant.simulate import simulate_child, simulate_rebuild
from syncytium.quant.stagger import child_codes, child_weights, offset, quantize_children

__all__ = [
    "ChannelScales",
    "Estimator",
    "Grid",
    "Scheme",
    "child_codes",
    "child_weights",
    "fit_grid",
    "from_groups",
    "mean_weights",
    "normalize",
    "offset",
    "parent_codes",
    "parent_weights",
    "partial_weights",
    "quantize_children",
    "search_scales",
    "simulate_child",
    "simulate_rebuild",
    "to_groups",
]
