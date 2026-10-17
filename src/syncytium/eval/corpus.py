"""Evaluation text."""

from datasets import load_dataset
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def load_wikitext2(tokenizer: PreTrainedTokenizerBase, split: str = "test") -> Tensor:
    """Token ids of a WikiText-2 split as one 1-D tensor.

    Documents are joined with blank lines, the convention used by most quantization papers,
    so results are comparable with published numbers.
    """
    text = "\n\n".join(load_dataset(*WIKITEXT, split=split)["text"])
    # The corpus is longer than the model's context; it is split into windows afterwards.
    return tokenizer(text, return_tensors="pt", verbose=False).input_ids[0]
