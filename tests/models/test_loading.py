import torch

from syncytium.models import load_model


def test_load_model_returns_frozen_model_in_eval_mode(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    model = load_model(str(tmp_path), device="cpu", dtype=torch.float32)
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
