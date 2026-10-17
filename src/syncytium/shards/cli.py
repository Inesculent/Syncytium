"""Command lines for shard files.

``syncytium-shard MODEL --out DIR [scheme options]`` writes the children;
``syncytium-rebuild DIR [--children K ...] [--repeat R]`` times rebuilds from them.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from syncytium.models import load_model, split_parameters
from syncytium.quant import Scheme
from syncytium.shards.manifest import shard_name
from syncytium.shards.timing import log_timing, time_rebuild
from syncytium.shards.writer import write_children


def main(argv: Sequence[str] | None = None) -> None:
    """Split a model into N child shards plus a manifest."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model", help="Hugging Face model id or local path")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--children", type=int, default=4)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--rotation-seed", type=int, default=None)
    args = parser.parse_args(argv)
    scheme = Scheme(args.bits, args.group_size, args.children, args.phase, args.rotation_seed)
    model = load_model(args.model, device="cpu", dtype=torch.bfloat16)
    write_children(*split_parameters(model), scheme, args.model, args.out)
    for index in range(scheme.num_children):
        size = (args.out / shard_name(index)).stat().st_size
        print(f"{shard_name(index)}  {size / 1e9:.3f} GB")


def rebuild_main(argv: Sequence[str] | None = None) -> None:
    """Rebuild weights from shard files, time it and log the timing next to the shards."""
    parser = argparse.ArgumentParser(description=rebuild_main.__doc__)
    parser.add_argument("directory", type=Path, help="directory written by syncytium-shard")
    parser.add_argument("--children", type=int, nargs="+", help="child indices; default all")
    parser.add_argument("--repeat", type=int, default=1, help="number of timed rebuilds")
    args = parser.parse_args(argv)
    for _ in range(args.repeat):
        timing = time_rebuild(args.directory, args.children)
        log_timing(args.directory, timing)
        kind = "exact" if timing.exact else "partial"
        print(
            f"{kind} rebuild from {list(timing.children)}: {timing.bytes_read / 1e9:.3f} GB, "
            f"verify {timing.verify_seconds:.1f} s + rebuild {timing.rebuild_seconds:.1f} s"
        )
