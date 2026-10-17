import pytest
import torch

from syncytium.pool import PoolGridConfig

# Three children so that a stage can lose a device and still have more than one child, which
# is where the estimators differ. The tiny model's narrowest input dimension is 32.
GRID = PoolGridConfig(bits=4, group_size=32, children=3, stages=2, max_lost=3, samples=2)


@pytest.fixture
def grid():
    return GRID


@pytest.fixture
def batches():
    return torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
