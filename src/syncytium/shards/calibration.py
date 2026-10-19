"""Splitting a model into shards, with channel scales calibrated exactly as the sweeps do.

A scaled scheme (``docs/method.md``, "Optional channel scaling") needs scales searched from
the unquantized model's activations. They come from the sweeps' own
:func:`~syncytium.sweep.runner.calibrate` on WikiText-2 train windows, sized by the default
:class:`~syncytium.sweep.config.EvalConfig` (64 windows of 512 tokens), and each weight's
search runs on its float32 copy, as a sweep's weight bank passes it.
"""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from syncytium.eval import load_wikitext2, windows
from syncytium.models import load_model, load_tokenizer, quantized_weights, split_parameters
from syncytium.quant import ChannelScales, Scheme
from syncytium.shards.manifest import Manifest
from syncytium.shards.writer import write_children
from syncytium.sweep.config import EvalConfig
from syncytium.sweep.runner import calibrate


def shard_model(name: str, scheme: Scheme, directory: Path, device: str = "cpu") -> Manifest:
    """Load model ``name`` in bf16 and write it as N children, calibrating scales if needed.

    Args:
        name: Hugging Face model id or local path, recorded in the manifest.
        scheme: Shared quantization settings; a scaled scheme is calibrated first.
        directory: Output directory.
        device: Where the model is loaded and calibrated.
    """
    model = load_model(name, device=device, dtype=torch.bfloat16)
    scales = None
    if scheme.scaling:
        scales = calibrated_scales(model, load_tokenizer(name), scheme)
    return write_children(*split_parameters(model), scheme, name, directory, scales)


def calibrated_scales(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, scheme: Scheme
) -> dict[str, Tensor]:
    """Channel scales of ``model``'s quantized weights for ``scheme``, as a sweep finds them.

    Weights that no calibrated linear layer reads have no statistics and are left out, so
    they stay unscaled, as in a sweep.
    """
    evaluation = EvalConfig()
    train = load_wikitext2(tokenizer, split="train")
    batches = windows(train, evaluation.calibration_window, evaluation.calibration_windows)
    return weight_scales(calibrate(model, batches), quantized_weights(model), scheme)


def weight_scales(
    searched: ChannelScales, weights: Mapping[str, Tensor], scheme: Scheme
) -> dict[str, Tensor]:
    """The scales ``searched`` gives each weight (in float32 on its device), where it has any."""
    scales = {}
    for name, weight in weights.items():
        vector = searched(name, weight.detach().float(), scheme)
        if vector is not None:
            scales[name] = vector
    return scales
