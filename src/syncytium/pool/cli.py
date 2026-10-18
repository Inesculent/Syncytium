"""Command line: ``syncytium-pool run CONFIG`` and ``syncytium-pool report RESULTS_DIR``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from syncytium.pool.config import load_pool_config
from syncytium.pool.runner import run_pool
from syncytium.pool.summary import write_pool_report


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate device losses in a pooled pipeline, or summarize a finished sweep."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate the device states in a config")
    run.add_argument("config", type=Path, help="path to a pooled config in configs/")
    report = commands.add_parser("report", help="write the summary of a finished sweep")
    report.add_argument("results", type=Path, help="sweep output directory")
    report.add_argument("--out", type=Path, help="defaults to docs/results/<sweep name>")
    args = parser.parse_args(argv)
    if args.command == "run":
        # Third-party libraries stay at WARNING; only this package reports progress.
        logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
        logging.getLogger("syncytium").setLevel(logging.INFO)
        run_pool(load_pool_config(args.config), args.config)
    else:
        write_pool_report(args.results, args.out or Path("docs/results") / args.results.name)


if __name__ == "__main__":
    main()
