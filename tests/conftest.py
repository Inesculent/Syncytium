import pytest
import torch


@pytest.fixture
def groups() -> torch.Tensor:
    """Weight-like values already arranged in groups of 64, in float64 for exact checks."""
    generator = torch.Generator().manual_seed(0)
    return 0.02 * torch.randn(512, 64, generator=generator, dtype=torch.float64)
