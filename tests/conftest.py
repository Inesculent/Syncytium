import pytest
import torch
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@pytest.fixture
def groups() -> torch.Tensor:
    """Weight-like values already arranged in groups of 64, in float64 for exact checks."""
    generator = torch.Generator().manual_seed(0)
    return 0.02 * torch.randn(512, 64, generator=generator, dtype=torch.float64)


@pytest.fixture(scope="session")
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


PAD, EOS = 0, 1


class ByteTokenizer:
    """A tokenizer for the tiny test model: one token per byte, with 0 and 1 reserved.

    Answer letters " A" to " D" are single tokens, as they are for Qwen3.5.
    """

    pad_token_id = PAD
    eos_token_id = EOS

    def encode(self, text, add_special_tokens=False):
        if text in (" A", " B", " C", " D"):
            return [ord(text[1])]
        return [max(b, 2) for b in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(i for i in ids if i > 1).decode("utf-8", errors="replace")


@pytest.fixture
def tokenizer():
    return ByteTokenizer()
