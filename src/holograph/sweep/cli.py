"""Command line: ``holograph-sweep CONFIG``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from holograph.sweep.config import load_config
from holograph.sweep.runner import run_sweep


def main(argv: Sequence[str] | None = None) -> None:
    """Run the perplexity sweep described by a TOML config."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=Path, help="path to a sweep config in configs/")
    args = parser.parse_args(argv)
    # Third-party libraries stay at WARNING; only this package reports progress.
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("holograph").setLevel(logging.INFO)
    run_sweep(load_config(args.config), args.config)
