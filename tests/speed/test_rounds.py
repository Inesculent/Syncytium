import torch

from syncytium.speed import trace_rounds


def test_four_rounds_per_layer_plus_embedding_and_head(tiny_model, tiny_config):
    rounds = trace_rounds(tiny_model)
    assert len(rounds) == 4 * tiny_config.num_hidden_layers + 2
    assert rounds[0].layers == ("model.embed_tokens",)
    assert rounds[-1].head
    assert rounds[-1].values == tiny_config.vocab_size
    assert not any(r.head for r in rounds[:-1])


def test_layers_reading_one_input_share_a_round(tiny_model):
    rounds = trace_rounds(tiny_model)
    names = [r.layers for r in rounds]
    assert ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj") in names
    attention = [r for r in rounds if any(".self_attn.q_proj" in n for n in r.layers)]
    assert len(attention) == 1
    assert {n.rsplit(".", 1)[1] for n in attention[0].layers} == {"q_proj", "k_proj", "v_proj"}
    linear = rounds[1]
    assert all(".linear_attn.in_proj" in n for n in linear.layers)
    assert len(linear.layers) > 1


def test_round_widths_are_the_layers_output_widths(tiny_model):
    widths = {
        name: module.out_features
        for name, module in tiny_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    for r in trace_rounds(tiny_model)[1:]:
        assert r.values == sum(widths[name] for name in r.layers)


def test_tracing_works_without_weights(tiny_config):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tiny_config)
    assert len(trace_rounds(model)) == 4 * tiny_config.num_hidden_layers + 2
