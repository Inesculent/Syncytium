"""Weight memory of children, rebuilt parents and dense models (``docs/method.md``).

Footprints are computed from parameter counts. Each quantization group stores its minimum
m and step Δ in fp16; parameters that are not quantized stay in bf16.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn

from holograph.quant import Scheme

SCALE_BITS = 16
DENSE_BITS = 16


@dataclass(frozen=True)
class ParameterCounts:
    """Parameters of a model, split by whether they are quantized."""

    quantized: int
    full_precision: int

    @property
    def total(self) -> int:
        """All parameters."""
        return self.quantized + self.full_precision


@dataclass(frozen=True)
class Footprint:
    """Weight memory of one model representation."""

    total_bits: float
    params: int

    @property
    def gigabytes(self) -> float:
        """Size in GB (10^9 bytes)."""
        return self.total_bits / 8e9

    @property
    def bits_per_weight(self) -> float:
        """Average bits per parameter, scales and unquantized tensors included."""
        return self.total_bits / self.params

    @property
    def reduction(self) -> float:
        """How many times smaller than the same model in bf16."""
        return DENSE_BITS / self.bits_per_weight


def count_parameters(model: nn.Module, quantized: Mapping[str, nn.Parameter]) -> ParameterCounts:
    """Count quantized and full-precision parameters, counting tied weights once."""
    quantized_ids = {id(p) for p in quantized.values()}
    sizes = {id(p): p.numel() for p in model.parameters()}  # parameters() skips tied copies
    in_quantized = sum(size for key, size in sizes.items() if key in quantized_ids)
    full_precision = sum(sizes.values()) - in_quantized
    return ParameterCounts(quantized=in_quantized, full_precision=full_precision)


def code_bits_of_parent(scheme: Scheme) -> int:
    """Bits per parent code: Q ranges over [0, N (2^b - 1)]."""
    return math.ceil(math.log2(scheme.num_children * ((1 << scheme.bits) - 1) + 1))


def child_footprint(counts: ParameterCounts, scheme: Scheme) -> Footprint:
    """Weight memory of one child."""
    return _quantized_footprint(counts, scheme, code_bits=scheme.bits)


def parent_footprint(counts: ParameterCounts, scheme: Scheme) -> Footprint:
    """Weight memory of the parent rebuilt from all children, stored as integer codes."""
    return _quantized_footprint(counts, scheme, code_bits=code_bits_of_parent(scheme))


def dense_footprint(counts: ParameterCounts) -> Footprint:
    """Weight memory of the unquantized model in bf16."""
    return Footprint(total_bits=counts.total * DENSE_BITS, params=counts.total)


def _quantized_footprint(counts: ParameterCounts, scheme: Scheme, code_bits: int) -> Footprint:
    if counts.quantized % scheme.group_size:
        raise ValueError(
            f"{counts.quantized} quantized parameters do not divide into groups of "
            f"{scheme.group_size}"
        )
    groups = counts.quantized // scheme.group_size
    bits = (
        counts.quantized * code_bits + groups * 2 * SCALE_BITS + counts.full_precision * DENSE_BITS
    )
    return Footprint(total_bits=bits, params=counts.total)
