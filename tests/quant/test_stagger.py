import pytest
import torch

from holograph.quant import (
    child_codes,
    child_weights,
    fit_grid,
    normalize,
    offset,
    quantize_children,
)

BITS = [2, 3, 4]
CHILDREN = [1, 2, 4, 8]
TOLERANCE = 1e-9


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", CHILDREN)
def test_codes_fit_in_bits(groups, bits, num_children):
    grid = fit_grid(groups, bits)
    for codes in quantize_children(groups, grid, num_children):
        assert codes.dtype == torch.uint8
        assert int(codes.max()) <= 2**bits - 1


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", CHILDREN)
def test_child_error_is_at_most_half_a_step(groups, bits, num_children):
    grid = fit_grid(groups, bits)
    for index, codes in enumerate(quantize_children(groups, grid, num_children)):
        error = (child_weights(codes, grid, index, num_children) - groups).abs()
        assert torch.all(error <= grid.step * (0.5 + TOLERANCE))


def test_children_differ_from_each_other(groups):
    first, second = quantize_children(groups, fit_grid(groups, 4), num_children=2)
    assert not torch.equal(first, second)


def test_middle_child_is_round_to_nearest(groups):
    grid = fit_grid(groups, 4)
    x = normalize(groups, grid)
    codes = child_codes(x, grid, index=2, num_children=4)
    expected = grid.minimum + grid.step * torch.floor(x + 0.5)
    assert torch.allclose(child_weights(codes, grid, index=2, num_children=4), expected)


@pytest.mark.parametrize("index", [-1, 4])
def test_offset_rejects_out_of_range_index(index):
    with pytest.raises(ValueError, match="child index"):
        offset(index, num_children=4)
