import pytest
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@pytest.fixture
def tiny_config() -> Qwen3_5TextConfig:
    """A four-layer Qwen3.5 text config with both Gated DeltaNet and full attention layers."""
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_model(tiny_config) -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM(tiny_config).eval()
