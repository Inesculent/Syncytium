"""Command line: ``syncytium-runs [--pending] [--kind KIND]``."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from syncytium.runs.status import Kind, State, inventory, table


def main(argv: Sequence[str] | None = None) -> None:
    """List every config's run: what it is, how far it got, and whether it is reported."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--root", type=Path, default=Path(), help="repository root")
    parser.add_argument("--pending", action="store_true", help="only runs not complete here")
    parser.add_argument("--kind", type=Kind, choices=list(Kind), help="only runs of this kind")
    args = parser.parse_args(argv)
    runs, strays = inventory(args.root)
    if args.pending:
        runs = [run for run in runs if run.state != State.COMPLETE]
    if args.kind:
        runs = [run for run in runs if run.kind == args.kind]
    print(table(runs, strays))


if __name__ == "__main__":
    main()
