import pytest
import torch

from syncytium.quant import ChannelScales, Scheme, search_scales, simulate_child, simulate_rebuild
from syncytium.quant.scaling import ALPHAS, candidate_scales


@pytest.fixture
def weight(groups):
    return groups.reshape(64, 512)


@pytest.fixture
def moments():
    # Activation power varies over input channels, with a few loud ones, as in real layers.
    generator = torch.Generator().manual_seed(0)
    power = torch.rand(512, generator=generator, dtype=torch.float64) + 0.1
    power[::64] *= 400
    return power


def weighted_error(weight, moments, scheme, scales):
    return float(((simulate_child(weight, scheme, 0, scales) - weight) ** 2 * moments).sum())


def test_unit_scales_change_nothing(weight):
    scheme = Scheme(3, 64, 4, 0.5)
    ones = torch.ones(512, dtype=weight.dtype)
    assert torch.equal(simulate_child(weight, scheme, 1, ones), simulate_child(weight, scheme, 1))


@pytest.mark.parametrize("rotation_seed", [None, 0])
def test_rebuild_from_scaled_children_is_still_their_mean(weight, moments, rotation_seed):
    scheme = Scheme(3, 64, 4, 0.5, rotation_seed, scaling=True)
    scales = search_scales(weight, moments, scheme)
    mean = torch.stack([simulate_child(weight, scheme, k, scales) for k in range(4)]).mean(0)
    assert torch.allclose(simulate_rebuild(weight, scheme, range(4), scales=scales), mean)


def test_candidate_scales_are_normalized_and_follow_activation_power(moments):
    scales = candidate_scales(moments, 0.5)
    assert float(scales.max() * scales.min()) == pytest.approx(1.0)
    assert torch.equal(candidate_scales(moments, 0.0), torch.ones_like(moments))
    assert scales[0] > scales[1]  # channel 0 is loud


def test_search_never_does_worse_than_no_scaling(weight, moments):
    scheme = Scheme(3, 64, 1, 0.5)
    ones = torch.ones(512, dtype=weight.dtype)
    searched = search_scales(weight, moments, scheme)
    assert weighted_error(weight, moments, scheme, searched) <= weighted_error(
        weight, moments, scheme, ones
    )


def test_search_scales_loud_channels_up(weight, moments):
    searched = search_scales(weight, moments, Scheme(3, 64, 1, 0.5))
    assert not torch.equal(searched, torch.ones_like(searched))
    loud, quiet = searched[::64].mean(), searched[1::64].mean()
    assert loud > quiet


def test_search_ignores_n_and_phase(weight, moments):
    first = search_scales(weight, moments, Scheme(3, 64, 4, 0.0))
    assert torch.equal(first, search_scales(weight, moments, Scheme(3, 64, 8, 0.5)))


def test_channel_scales_cache_and_skip_weights_without_statistics(weight, moments):
    scales = ChannelScales({"loud": moments})
    scheme = Scheme(3, 64, 4, scaling=True)
    first = scales("loud", weight, scheme)
    assert scales("loud", weight, Scheme(3, 64, 8, 0.5, scaling=True)) is first
    assert scales("other", weight, scheme) is None


def test_alpha_grid_includes_no_scaling():
    assert ALPHAS[0] == 0.0 and ALPHAS[-1] == 1.0
