"""Running a sweep: evaluate each distinct condition once and log every condition."""

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel

from syncytium.eval import (
    Footprint,
    ParameterCounts,
    child_footprint,
    count_parameters,
    dense_footprint,
    load_wikitext2,
    parent_footprint,
    perplexity,
    windows,
)
from syncytium.models import (
    WeightBank,
    input_moments,
    load_model,
    load_tokenizer,
    missing_moments,
    quantized_weights,
)
from syncytium.quant import ChannelScales, Scheme
from syncytium.sweep.conditions import Condition, Kind, plan
from syncytium.sweep.config import SweepConfig
from syncytium.sweep.records import ResultLog, append_run_metadata

logger = logging.getLogger(__name__)


def run_sweep(config: SweepConfig, config_path: Path) -> None:
    """Load the model and corpus named in ``config`` and evaluate its grid."""
    append_run_metadata(config.output / "runs.jsonl", config_path)
    model = load_model(config.model.path, config.model.device, getattr(torch, config.model.dtype))
    tokenizer = load_tokenizer(config.model.path)
    evaluation = config.evaluation
    batches = windows(load_wikitext2(tokenizer), evaluation.window, evaluation.max_windows)
    log = ResultLog(config.output / "results.jsonl")
    conditions = plan(config.grid)
    scales = None
    if any(c.scheme is not None and c.scheme.scaling for c in conditions):
        train = load_wikitext2(tokenizer, split="train")
        calibration = windows(train, evaluation.calibration_window, evaluation.calibration_windows)
        scales = calibrate(model, calibration)
    run_conditions(conditions, model, batches, log, scales)


def calibrate(model: PreTrainedModel, batches: Tensor) -> ChannelScales:
    """Channel scales from the unquantized model's activations on ``batches``."""
    moments = input_moments(model, batches)
    count, length = batches.shape
    unscaled = missing_moments(moments, model) or "none"
    logger.info("calibrated on %d windows of %d tokens; unscaled: %s", count, length, unscaled)
    return ChannelScales(moments)


def run_conditions(
    conditions: Sequence[Condition],
    model: PreTrainedModel,
    batches: Tensor,
    log: ResultLog,
    scales: ChannelScales | None = None,
) -> None:
    """Evaluate ``conditions``, skipping ones already logged and reusing equal-key results.

    ``scales`` are needed when a condition scales its channels (see :func:`calibrate`).
    """
    weights = quantized_weights(model)
    bank = WeightBank(weights)
    counts = count_parameters(model, weights)
    done = log.completed()
    by_key = {c.key: done[c.id] for c in conditions if c.id in done}
    for position, condition in enumerate(conditions, start=1):
        if condition.id in done:
            continue
        source = by_key.get(condition.key)
        if source is None:
            row = _evaluate(condition, bank, model, batches, counts, scales)
            by_key[condition.key] = row
        else:
            row = _row(condition, source["perplexity"], 0.0, counts, reused_from=source["id"])
        log.append(row)
        logger.info("[%d/%d] %s  ppl %.3f", position, len(conditions), row["id"], row["perplexity"])


def _evaluate(
    condition: Condition,
    bank: WeightBank,
    model: PreTrainedModel,
    batches: Tensor,
    counts: ParameterCounts,
    scales: ChannelScales | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    transform = condition.transform(scales)
    if transform is None:
        bank.restore()
    else:
        bank.apply_named(transform)
    value = perplexity(model, batches)
    return _row(condition, value, time.perf_counter() - start, counts, reused_from=None)


def _row(
    condition: Condition,
    value: float,
    seconds: float,
    counts: ParameterCounts,
    reused_from: str | None,
) -> dict[str, Any]:
    scheme = condition.scheme
    settings = asdict(scheme) if scheme else dict.fromkeys(f.name for f in fields(Scheme))
    return {
        "id": condition.id,
        "kind": condition.kind.value,
        **settings,
        "indices": list(condition.indices),
        "offsets": sorted(str(o) for o in condition.offsets),
        "perplexity": value,
        "bits_per_weight": _footprint(condition, counts).bits_per_weight,
        "seconds": round(seconds, 2),
        "reused_from": reused_from,
    }


def _footprint(condition: Condition, counts: ParameterCounts) -> Footprint:
    if condition.scheme is None:
        return dense_footprint(counts)
    if condition.kind in (Kind.CHILD, Kind.RTN):
        return child_footprint(counts, condition.scheme)
    return parent_footprint(counts, condition.scheme)
