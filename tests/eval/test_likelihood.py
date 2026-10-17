import math

import pytest
import torch

from holograph.eval import likelihood, perplexity, windows


def test_windows_drop_the_remainder():
    batches = windows(torch.arange(10), length=3)
    assert batches.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_windows_respect_the_limit():
    assert windows(torch.arange(10), length=3, limit=2).shape == (2, 3)


def test_windows_reject_short_input():
    with pytest.raises(ValueError, match="shorter than one window"):
        windows(torch.arange(2), length=3)


def test_matches_the_model_loss(tiny_model):
    batches = torch.randint(0, 256, (3, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        losses = [tiny_model(row[None], labels=row[None]).loss for row in batches]
    assert perplexity(tiny_model, batches) == pytest.approx(math.exp(torch.stack(losses).mean()))


def test_chunked_output_head_gives_the_same_result(tiny_model, monkeypatch):
    batches = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(1))
    whole = perplexity(tiny_model, batches)
    monkeypatch.setattr(likelihood, "HEAD_CHUNK", 5)
    assert perplexity(tiny_model, batches) == pytest.approx(whole)
