"""Loading the text stack of a causal language model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def load_model(
    name_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> PreTrainedModel:
    """Load a causal language model for evaluation.

    For multimodal checkpoints such as Qwen3.5, ``AutoModelForCausalLM`` loads only the
    language model and skips the vision tower.
    """
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype, device_map=device)
    return model.eval().requires_grad_(False)


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer that matches :func:`load_model`."""
    return AutoTokenizer.from_pretrained(name_or_path)
