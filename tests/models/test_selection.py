from torch import nn

from syncytium.models import quantized_weights


def test_selects_embeddings_and_every_linear_layer(tiny_model):
    names = set(quantized_weights(tiny_model))
    linear = {f"{n}.weight" for n, m in tiny_model.named_modules() if isinstance(m, nn.Linear)}
    assert "model.embed_tokens.weight" in names
    assert linear - {"lm_head.weight"} <= names


def test_tied_output_head_is_listed_once(tiny_model):
    weights = quantized_weights(tiny_model)
    assert "lm_head.weight" not in weights
    assert len({id(w) for w in weights.values()}) == len(weights)


def test_keeps_small_sensitive_parameters_in_full_precision(tiny_model):
    names = quantized_weights(tiny_model)
    for kept in ("norm", "conv1d", "A_log", "dt_bias"):
        assert not any(kept in name for name in names)


def test_selected_weights_are_matrices(tiny_model):
    assert all(w.dim() == 2 for w in quantized_weights(tiny_model).values())
