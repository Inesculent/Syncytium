import pytest
import torch

from holograph.quant import fit_grid, from_groups, normalize, to_groups


def test_groups_round_trip():
    weight = torch.randn(8, 128)
    assert torch.equal(from_groups(to_groups(weight, 32), weight.shape), weight)


def test_to_groups_rejects_indivisible_dimension():
    with pytest.raises(ValueError, match="divisible"):
        to_groups(torch.randn(4, 30), 32)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_normalized_weights_span_the_grid(groups, bits):
    x = normalize(groups, fit_grid(groups, bits))
    assert torch.equal(x.amin(dim=1), torch.zeros(len(groups), dtype=x.dtype))
    assert torch.allclose(x.amax(dim=1), torch.full((len(groups),), 2.0**bits - 1, dtype=x.dtype))


def test_constant_group_has_positive_step():
    grid = fit_grid(torch.full((1, 64), 0.3, dtype=torch.float64), bits=4)
    assert grid.step.item() > 0


@pytest.mark.parametrize("bits", [0, 9])
def test_fit_grid_rejects_unsupported_bits(groups, bits):
    with pytest.raises(ValueError, match="bits"):
        fit_grid(groups, bits)
