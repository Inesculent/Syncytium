import torch

from syncytium.models import input_moments, missing_moments, quantized_weights


def batches():
    return torch.randint(0, 256, (3, 16), generator=torch.Generator().manual_seed(0))


def test_every_quantized_weight_gets_its_input_statistics(tiny_model):
    moments = input_moments(tiny_model, batches())
    weights = quantized_weights(tiny_model)
    assert missing_moments(moments, tiny_model) == []
    for name, weight in weights.items():
        assert moments[name].shape == (weight.shape[-1],)
        assert torch.all(moments[name] >= 0)


def test_tied_embedding_gets_the_output_heads_statistics(tiny_model):
    captured = []
    handle = tiny_model.lm_head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].flatten(0, -2).float())
    )
    try:
        moments = input_moments(tiny_model, batches())
    finally:
        handle.remove()
    inputs = torch.cat(captured)
    expected = inputs.square().mean(dim=0)
    assert torch.allclose(moments["model.embed_tokens.weight"], expected)


def test_hooks_are_removed(tiny_model):
    input_moments(tiny_model, batches())
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())
