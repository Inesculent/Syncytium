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


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_storage_grid_is_exact_in_fp16_and_still_covers_every_weight(groups, bits):
    grid = fit_grid(groups.float(), bits, storage=torch.float16)
    for values in (grid.minimum, grid.step):
        assert torch.equal(values.half().float(), values)
    x = normalize(groups.float(), grid)
    assert x.min() >= 0
    assert x.max() <= 2**bits - 1


def test_storage_grid_stays_close_to_the_exact_grid(groups):
    exact = fit_grid(groups.float(), 4)
    stored = fit_grid(groups.float(), 4, storage=torch.float16)
    assert torch.allclose(stored.step, exact.step, rtol=2e-3)


@pytest.mark.parametrize("bits", [0, 9])
def test_fit_grid_rejects_unsupported_bits(groups, bits):
    with pytest.raises(ValueError, match="bits"):
        fit_grid(groups, bits)
