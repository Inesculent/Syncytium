import pytest

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
