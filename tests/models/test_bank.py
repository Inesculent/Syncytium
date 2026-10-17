import torch

from holograph.models import WeightBank, quantized_weights


def snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_apply_writes_transformed_weights(tiny_model):
    weights = quantized_weights(tiny_model)
    WeightBank(weights).apply(torch.zeros_like)
    assert all(torch.count_nonzero(w) == 0 for w in weights.values())


def test_apply_leaves_unselected_parameters_alone(tiny_model):
    selected = {id(p) for p in quantized_weights(tiny_model).values()}
    before = snapshot(tiny_model)
    WeightBank(quantized_weights(tiny_model)).apply(torch.zeros_like)
    for name, param in tiny_model.named_parameters():
        if id(param) not in selected:
            assert torch.equal(param, before[name])


def test_conditions_do_not_compound(tiny_model):
    weights = quantized_weights(tiny_model)
    bank = WeightBank(weights)
    before = snapshot(tiny_model)
    bank.apply(lambda w: w + 1)
    bank.apply(lambda w: w + 1)
    name = "model.embed_tokens.weight"
    assert torch.allclose(weights[name], before[name] + 1)


def test_restore_returns_original_weights(tiny_model):
    bank = WeightBank(quantized_weights(tiny_model))
    before = snapshot(tiny_model)
    bank.apply(torch.zeros_like)
    bank.restore()
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
