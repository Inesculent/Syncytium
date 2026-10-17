"""Loading the text stack of a causal language model."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from syncytium.eval.memory import ParameterCounts, count_parameters
from syncytium.models.selection import quantized_weights


def load_model(
    name_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> PreTrainedModel:
    """Load a causal language model for evaluation.

    For multimodal checkpoints such as Qwen3.5, ``AutoModelForCausalLM`` loads only the
    language model and skips the vision tower.
    """
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype, device_map=device)
    return model.eval().requires_grad_(False)


def parameter_counts(name_or_path: str) -> ParameterCounts:
    """Quantized and full-precision parameter counts of a model, from its config alone.

    The model is built on the meta device, so no weights are downloaded or allocated.
    """
    config = AutoConfig.from_pretrained(name_or_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return count_parameters(model, quantized_weights(model))


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer that matches :func:`load_model`."""
    return AutoTokenizer.from_pretrained(name_or_path)
