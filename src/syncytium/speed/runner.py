"""Measuring each lone child's agreement with the fused group (H9; needs the model on a GPU).

Writes one row per child to ``acceptance.jsonl``: the fraction of positions where its greedy
choice matches the full group's, and the simulated tokens per speculative cycle for each
draft length. Everything else H9 needs is computed without the model by
:mod:`syncytium.speed.report`.
"""

import logging
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedModel

from syncytium.eval import load_wikitext2, windows
from syncytium.models import WeightBank, load_model, load_tokenizer, quantized_weights
from syncytium.quant import ChannelScales, Estimator
from syncytium.speed.acceptance import agreement_curve, greedy_choices
from syncytium.speed.config import SpeedConfig
from syncytium.sweep.records import ResultLog, append_run_metadata
from syncytium.sweep.runner import calibrate
from syncytium.tasks.conditions import Kind, TaskCondition

logger = logging.getLogger(__name__)

ACCEPTANCE = "acceptance.jsonl"


def run_speed(config: SpeedConfig, config_path: Path) -> None:
    """Load the model named in ``config`` and measure every child's agreement."""
    append_run_metadata(config.output / "runs.jsonl", config_path)
    model = load_model(config.model.path, config.model.device, getattr(torch, config.model.dtype))
    tokenizer = load_tokenizer(config.model.path)
    text = config.acceptance
    batches = windows(load_wikitext2(tokenizer), text.window, text.windows)
    scales = None
    if config.setting.scaling:
        evaluation = config.evaluation
        train = load_wikitext2(tokenizer, split="train")
        scales = calibrate(
            model, windows(train, evaluation.calibration_window, evaluation.calibration_windows)
        )
    measure_agreement(model, batches, config, scales)


def measure_agreement(
    model: PreTrainedModel,
    batches: Tensor,
    config: SpeedConfig,
    scales: ChannelScales | None = None,
) -> None:
    """Log each child's agreement with the full group on ``batches``, skipping logged ones."""
    setting, scheme = config.setting, config.setting.scheme
    log = ResultLog(config.output / ACCEPTANCE)
    done = log.completed()
    children = [TaskCondition(Kind.CHILD, (k,)) for k in range(setting.children)]
    if all(child.id in done for child in children):
        return
    bank = WeightBank(quantized_weights(model))
    everyone = TaskCondition(Kind.GROUP, tuple(range(setting.children)))
    bank.apply_named(everyone.transform(scheme, Estimator.MEAN, scales))
    fused = greedy_choices(model, batches)
    for child in children:
        if child.id in done:
            continue
        bank.apply_named(child.transform(scheme, Estimator.MEAN, scales))
        agree = greedy_choices(model, batches) == fused
        curve = agreement_curve(agree, config.acceptance.drafts)
        log.append(
            {
                "id": child.id,
                "child": child.indices[0],
                "agreement": agree.float().mean().item(),
                "positions": agree.numel(),
                "tokens_per_cycle": {str(k): v for k, v in curve.items()},
            }
        )
        logger.info("%s agrees with the group at %.3f of positions", child.id, agree.float().mean())
