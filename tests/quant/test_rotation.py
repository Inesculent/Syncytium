import pytest
import torch

from holograph.quant import Scheme, simulate_child, simulate_rebuild
from holograph.quant.rotation import hadamard, random_rotation


def test_hadamard_of_order_two():
    assert hadamard(2).tolist() == [[1, 1], [1, -1]]


@pytest.mark.parametrize("size", [1, 4, 64])
def test_hadamard_rows_are_orthogonal(size):
    matrix = hadamard(size)
    assert torch.equal(matrix @ matrix.T, size * torch.eye(size))


@pytest.mark.parametrize("size", [0, 3, 48])
def test_hadamard_rejects_other_sizes(size):
    with pytest.raises(ValueError, match="power of two"):
        hadamard(size)


def test_random_rotation_is_orthogonal():
    rotation = random_rotation(64, seed=0)
    assert torch.allclose(rotation @ rotation.T, torch.eye(64), atol=1e-6)


def test_random_rotation_depends_only_on_the_seed():
    assert torch.equal(random_rotation(64, seed=3), random_rotation(64, seed=3))
    assert not torch.equal(random_rotation(64, seed=3), random_rotation(64, seed=4))


@pytest.fixture
def weight(groups):
    return groups.reshape(64, 512)


@pytest.mark.parametrize("phase", [0.0, 0.5])
def test_rebuild_still_equals_mean_of_children(weight, phase):
    scheme = Scheme(3, 64, 4, phase=phase, rotation_seed=0)
    mean = torch.stack([simulate_child(weight, scheme, k) for k in range(4)]).mean(dim=0)
    assert torch.allclose(simulate_rebuild(weight, scheme, range(4)), mean)


def test_rotation_preserves_the_group_error_bound(weight):
    scheme = Scheme(3, 64, 4, rotation_seed=0)
    error = (simulate_rebuild(weight, scheme, range(4)) - weight).reshape(-1, 64)
    rotated = weight.reshape(-1, 64) @ random_rotation(64, 0).double()
    step = (rotated.amax(1) - rotated.amin(1)) / 7
    assert torch.all(error.norm(dim=1) <= 64**0.5 * step / 8 * (1 + 1e-9))


def mean_squared_error(weight, scheme):
    return (simulate_child(weight, scheme, 0) - weight).pow(2).mean()


def test_rotation_helps_weights_with_outliers():
    torch.manual_seed(0)
    heavy_tailed = torch.distributions.StudentT(2.0).sample((256, 512)).double()
    plain = mean_squared_error(heavy_tailed, Scheme(3, 64, 1))
    rotated = mean_squared_error(heavy_tailed, Scheme(3, 64, 1, rotation_seed=0))
    assert rotated < plain


def test_scheme_rejects_rotation_with_other_group_sizes():
    with pytest.raises(ValueError, match="power-of-two"):
        Scheme(4, 48, 4, rotation_seed=0)
