"""Find where a live run's shard sets first give non-finite numbers (``python -m ...``).

The first 4B development run of H18 gave NaN logits everywhere: in the live members, in the
baseline's stages and in the simulation reference built from the children's files. This walks
the chain one step at a time on one GPU and prints one JSON line per step:

1. the bf16 parent itself, on the first probe prompt;
2. the channel scales file of each set;
3. the weights rebuilt from each set's files, against the parent's;
4. the parent model holding the rebuilt weights;
5. one packed member built from each set, as a live device builds it.
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from syncytium.live.packed import load_member
from syncytium.live.run import LiveConfig, problems
from syncytium.models import load_model, load_tokenizer
from syncytium.shards import MANIFEST_NAME, SCALES_NAME, Manifest, shard_name
from syncytium.shards.reader import read_scales, rebuild_from_files
from syncytium.tasks import gsm8k


def nonfinite(tensors: Mapping[str, Tensor]) -> dict[str, int]:
    """How many non-finite values each tensor holds, for the tensors that hold any."""
    counts = {name: int((~torch.isfinite(t.float())).sum()) for name, t in tensors.items()}
    return {name: n for name, n in counts.items() if n}


def scale_report(scales: Mapping[str, Tensor]) -> dict[str, Any]:
    """The extremes of a set's channel scales, and the vectors with zero or non-finite ones."""
    if not scales:
        return {"scaled": False}
    every = torch.cat([s.float().flatten() for s in scales.values()])
    finite = every[torch.isfinite(every)]
    zero = [name for name, s in scales.items() if bool((s == 0).any())]
    return {
        "scaled": True,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "nonfinite": nonfinite(scales),
        "zero": zero,
    }


def weight_gaps(rebuilt: Mapping[str, Tensor], parent: Mapping[str, Tensor]) -> list[list[Any]]:
    """The five rebuilt weights farthest from the parent's, as ``[name, largest gap]``."""
    gaps = []
    for name, w in rebuilt.items():
        if name in parent:
            gap = float((w.float() - parent[name].float().cpu()).abs().max())
            gaps.append([name, float("inf") if math.isnan(gap) else gap])  # NaN sorts first
    return sorted(gaps, key=lambda g: -g[1])[:5]


def logits_report(model: Any, ids: Sequence[int]) -> dict[str, Any]:
    """Whether ``model``'s logits on ``ids`` are finite, and its last top-1 token."""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([list(ids)], device=device)).logits[0].float()
    bad = ~torch.isfinite(logits)
    return {
        "finite": not bool(bad.any()),
        "nonfinite_fraction": float(bad.float().mean()),
        "positions_affected": int(bad.any(dim=-1).sum()),
        "last_top1": int(logits[-1].argmax()),
    }


def _say(step: str, **fields: Any) -> None:
    print(json.dumps({"step": step, **fields}, default=str), flush=True)


def _check_set(name: str, directory: Path, model: Any, ids: list[int], device: str) -> None:
    manifest = Manifest.load(directory / MANIFEST_NAME)
    _say(f"{name}: scales", **scale_report(read_scales(directory / SCALES_NAME, manifest)))
    shards = {k: directory / shard_name(k) for k in range(manifest.scheme.num_children)}
    rebuilt = rebuild_from_files(shards, manifest, torch.bfloat16)
    parent = model.state_dict()
    gaps = weight_gaps(rebuilt, parent)
    _say(f"{name}: rebuilt weights", nonfinite=nonfinite(rebuilt), gaps=gaps)
    original = {k: v.clone() for k, v in parent.items() if k in rebuilt}
    model.load_state_dict(rebuilt, strict=False)
    report = logits_report(model, ids)
    _say(f"{name}: parent with rebuilt weights", loaded=len(original), of=len(rebuilt), **report)
    model.load_state_dict(original, strict=False)
    del rebuilt, original
    member = load_member(directory, 0, device)
    _say(f"{name}: packed member 0", **logits_report(member, ids))
    del member
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    """Report where a live config's shard sets first give non-finite numbers."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = LiveConfig.load(args.config)
    tokenizer = load_tokenizer(config.model)
    ids = tokenizer.encode(gsm8k.prompt(problems(config)[0][0].question))
    model = load_model(config.model, args.device)
    _say("parent", tokens=len(ids), **logits_report(model, ids))
    for name, directory in (("children", config.children), ("baseline", config.baseline)):
        _check_set(name, directory, model, ids, args.device)


if __name__ == "__main__":
    main()
