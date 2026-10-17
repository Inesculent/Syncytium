"""Command line: ``holograph-shard MODEL --out DIR [scheme options]``."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from holograph.models import load_model, split_parameters
from holograph.quant import Scheme
from holograph.shards.manifest import shard_name
from holograph.shards.writer import write_children


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
