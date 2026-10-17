"""Running a pooled-pipeline sweep: evaluate each distinct device state once."""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel

from syncytium.eval import count_parameters, dense_footprint, load_wikitext2, perplexity, windows
from syncytium.models import WeightBank, load_model, load_tokenizer, quantized_weights
from syncytium.pool.conditions import PoolCondition, plan
from syncytium.pool.config import PoolConfig, PoolGridConfig
from syncytium.pool.layout import assign_stages, device_footprints
from syncytium.quant import Scheme
from syncytium.sweep.records import ResultLog, append_run_metadata

logger = logging.getLogger(__name__)


def run_pool(config: PoolConfig, config_path: Path) -> None:
    """Load the model and corpus named in ``config`` and evaluate its device states."""
    append_run_metadata(config.output / "runs.jsonl", config_path)
    model = load_model(config.model.path, config.model.device, getattr(torch, config.model.dtype))
    ids = load_wikitext2(load_tokenizer(config.model.path))
    batches = windows(ids, config.evaluation.window, config.evaluation.max_windows)
    write_layout(config.output / "layout.json", model, config.pool)
    log = ResultLog(config.output / "results.jsonl")
    run_states(plan(config.pool), model, batches, log, config.pool.stages)


def run_states(
    conditions: Sequence[PoolCondition],
    model: PreTrainedModel,
    batches: Tensor,
    log: ResultLog,
    num_stages: int,
) -> None:
    """Evaluate ``conditions``, skipping ones already logged and reusing equal-key results."""
    weights = quantized_weights(model)
    bank = WeightBank(weights)
    stages = assign_stages(weights, num_stages)
    done = log.completed()
    by_key = {c.key: done[c.id] for c in conditions if c.id in done}
    for position, condition in enumerate(conditions, start=1):
        if condition.id in done:
            continue
        source = by_key.get(condition.key)
        if source is None:
            start = time.perf_counter()
            transform = condition.transform(stages)
            if transform is None:
                bank.restore()
            else:
                bank.apply_named(transform)
            value = perplexity(model, batches)
            row = _row(condition, value, time.perf_counter() - start, reused_from=None)
            by_key[condition.key] = row
        else:
            row = _row(condition, source["perplexity"], 0.0, reused_from=source["id"])
        log.append(row)
        logger.info("[%d/%d] %s  ppl %.3f", position, len(conditions), row["id"], row["perplexity"])


def write_layout(path: Path, model: PreTrainedModel, grid: PoolGridConfig) -> None:
    """Record the memory of one device on each stage next to that of the whole model."""
    counts = count_parameters(model, quantized_weights(model))
    devices = device_footprints(model, grid.scheme, grid.stages)
    layout = {
        "stages": grid.stages,
        "children": grid.children,
        "dense_gb": dense_footprint(counts).gigabytes,
        "device_gb": [footprint.gigabytes for footprint in devices],
        "device_params": [footprint.params for footprint in devices],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")


def _row(
    condition: PoolCondition, value: float, seconds: float, reused_from: str | None
) -> dict[str, Any]:
    scheme = condition.scheme
    settings = asdict(scheme) if scheme else dict.fromkeys(f.name for f in fields(Scheme))
    return {
        "id": condition.id,
        "kind": condition.kind.value,
        **settings,
        "estimator": condition.estimator.value if condition.estimator_matters else None,
        "alive": [sorted(alive) for alive in condition.alive],
        "lost": [list(device) for device in condition.lost],
        "perplexity": value,
        "seconds": round(seconds, 2),
        "reused_from": reused_from,
    }
