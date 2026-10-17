from itertools import pairwise

import pytest
import torch

from syncytium.quant import (
    child_weights,
    fit_grid,
    normalize,
    parent_codes,
    parent_weights,
    partial_weights,
    quantize_children,
)

BITS = [2, 3, 4]
CHILDREN = [1, 2, 4, 8]
PHASES = [0.0, 0.5]
TOLERANCE = 1e-9


def largest_offset_gap(indices, num_children, phase):
    """Largest gap between available offsets (k + phase)/N on the unit circle."""
    offsets = sorted((index + phase) / num_children for index in indices)
    gaps = [b - a for a, b in pairwise(offsets)]
    return max([*gaps, offsets[0] + 1 - offsets[-1]])


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", CHILDREN)
@pytest.mark.parametrize("phase", PHASES)
def test_parent_codes_satisfy_hermite_identity(groups, bits, num_children, phase):
    grid = fit_grid(groups, bits)
    codes = parent_codes(quantize_children(groups, grid, num_children, phase))
    expected = torch.floor(num_children * normalize(groups, grid) + phase).to(torch.int32)
    assert torch.equal(codes, expected)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", CHILDREN)
@pytest.mark.parametrize("phase", PHASES)
def test_parent_error_is_at_most_half_a_fine_step(groups, bits, num_children, phase):
    grid = fit_grid(groups, bits)
    codes = parent_codes(quantize_children(groups, grid, num_children, phase))
    error = (parent_weights(codes, grid, num_children, phase) - groups).abs()
    assert torch.all(error <= grid.step / num_children * (0.5 + TOLERANCE))


@pytest.mark.parametrize("num_children", CHILDREN)
@pytest.mark.parametrize("phase", PHASES)
def test_parent_equals_mean_of_children(groups, num_children, phase):
    grid = fit_grid(groups, 4)
    children = quantize_children(groups, grid, num_children, phase)
    mean = torch.stack(
        [child_weights(c, grid, k, num_children, phase) for k, c in enumerate(children)]
    ).mean(dim=0)
    parent = parent_weights(parent_codes(children), grid, num_children, phase)
    assert torch.allclose(parent, mean)


@pytest.mark.parametrize("num_children", CHILDREN)
@pytest.mark.parametrize("phase", PHASES)
def test_partial_rebuild_with_every_child_equals_parent(groups, num_children, phase):
    grid = fit_grid(groups, 4)
    children = quantize_children(groups, grid, num_children, phase)
    partial = partial_weights(dict(enumerate(children)), grid, num_children, phase)
    parent = parent_weights(parent_codes(children), grid, num_children, phase)
    assert torch.allclose(partial, parent)


@pytest.mark.parametrize(
    ("num_children", "indices"),
    [(4, (0,)), (4, (1, 2, 3)), (4, (0, 2)), (4, (0, 1, 3)), (8, (3, 7)), (8, (0, 1, 2, 5, 6))],
)
@pytest.mark.parametrize("phase", PHASES)
def test_partial_error_is_bounded_by_largest_offset_gap(groups, num_children, indices, phase):
    grid = fit_grid(groups, 3)
    children = quantize_children(groups, grid, num_children, phase)
    available = {index: children[index] for index in indices}
    error = (partial_weights(available, grid, num_children, phase) - groups).abs()
    bound = largest_offset_gap(indices, num_children, phase) / 2
    assert torch.all(error <= grid.step * (bound + TOLERANCE))


def test_partial_rebuild_from_one_child_equals_that_child(groups):
    grid = fit_grid(groups, 4)
    children = quantize_children(groups, grid, num_children=4)
    partial = partial_weights({3: children[3]}, grid, num_children=4)
    assert torch.allclose(partial, child_weights(children[3], grid, index=3, num_children=4))


def test_rebuild_requires_at_least_one_child(groups):
    grid = fit_grid(groups, 4)
    with pytest.raises(ValueError, match="at least one child"):
        parent_codes([])
    with pytest.raises(ValueError, match="at least one child"):
        partial_weights({}, grid, num_children=4)
