import pytest

from syncytium.eval import (
    ParameterCounts,
    child_footprint,
    count_parameters,
    dense_footprint,
    parent_footprint,
)
from syncytium.models import quantized_weights
from syncytium.quant import Scheme

ALL_QUANTIZED = ParameterCounts(quantized=64_000, full_precision=0)


@pytest.mark.parametrize(("bits", "expected"), [(4, 4.5), (3, 3.5), (2, 2.5)])
def test_child_adds_half_a_bit_of_scales_at_group_size_64(bits, expected):
    footprint = child_footprint(ALL_QUANTIZED, Scheme(bits, group_size=64, num_children=4))
    assert footprint.bits_per_weight == pytest.approx(expected)


@pytest.mark.parametrize(("num_children", "code_bits"), [(2, 5), (4, 6), (8, 7), (16, 8)])
def test_parent_needs_log2_n_extra_bits(num_children, code_bits):
    footprint = parent_footprint(ALL_QUANTIZED, Scheme(4, group_size=64, num_children=num_children))
    assert footprint.bits_per_weight == pytest.approx(code_bits + 0.5)


def test_full_precision_parameters_stay_in_bf16():
    counts = ParameterCounts(quantized=64, full_precision=64)
    footprint = child_footprint(counts, Scheme(4, group_size=64, num_children=4))
    assert footprint.total_bits == 64 * 4 + 32 + 64 * 16


def test_dense_model_has_no_reduction():
    assert dense_footprint(ALL_QUANTIZED).reduction == 1


def test_group_size_must_divide_quantized_parameters():
    with pytest.raises(ValueError, match="groups of 64"):
        child_footprint(ParameterCounts(100, 0), Scheme(4, group_size=64, num_children=4))


def test_counts_tied_weights_once(tiny_model):
    counts = count_parameters(tiny_model, quantized_weights(tiny_model))
    assert counts.total == sum(p.numel() for p in tiny_model.parameters())
    assert counts.quantized == sum(p.numel() for p in quantized_weights(tiny_model).values())
