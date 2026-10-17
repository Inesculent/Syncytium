import pytest
import torch

from syncytium.quant import (
    Scheme,
    child_weights,
    fit_grid,
    partial_weights,
    quantize_children,
    simulate_child,
    simulate_rebuild,
    to_groups,
)
from syncytium.quant import simulate as simulate_module

SCHEME = Scheme(bits=3, group_size=64, num_children=4)


@pytest.fixture
def weight(groups):
    return groups.reshape(64, 512)


def test_child_matches_group_level_functions(weight):
    groups = to_groups(weight, SCHEME.group_size)
    grid = fit_grid(groups, SCHEME.bits)
    codes = quantize_children(groups, grid, SCHEME.num_children)[1]
    expected = child_weights(codes, grid, 1, SCHEME.num_children).reshape(weight.shape)
    assert torch.equal(simulate_child(weight, SCHEME, index=1), expected)


def test_full_rebuild_matches_partial_formula_with_every_child(weight):
    groups = to_groups(weight, SCHEME.group_size)
    grid = fit_grid(groups, SCHEME.bits)
    children = dict(enumerate(quantize_children(groups, grid, SCHEME.num_children)))
    expected = partial_weights(children, grid, SCHEME.num_children).reshape(weight.shape)
    assert torch.allclose(simulate_rebuild(weight, SCHEME, indices=range(4)), expected)


def test_rebuild_error_is_at_most_half_a_fine_step(weight):
    groups = to_groups(weight, SCHEME.group_size)
    step = fit_grid(groups, SCHEME.bits).step
    rebuilt = to_groups(simulate_rebuild(weight, SCHEME, indices=range(4)), SCHEME.group_size)
    assert torch.all((rebuilt - groups).abs() <= step / 4 * (0.5 + 1e-9))


def test_missing_child_lowers_precision(weight):
    full = simulate_rebuild(weight, SCHEME, indices=range(4))
    partial = simulate_rebuild(weight, SCHEME, indices=[0, 1, 3])
    assert (partial - weight).abs().mean() > (full - weight).abs().mean()


def test_chunking_does_not_change_results(weight, monkeypatch):
    unchunked = simulate_rebuild(weight, SCHEME, indices=[0, 2])
    monkeypatch.setattr(simulate_module, "CHUNK_GROUPS", 7)
    assert torch.equal(simulate_rebuild(weight, SCHEME, indices=[0, 2]), unchunked)
