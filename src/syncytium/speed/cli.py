"""Command line: ``syncytium-speed run CONFIG`` and ``syncytium-speed report CONFIG``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from syncytium.speed.config import load_speed_config
from syncytium.speed.report import write_report
from syncytium.speed.runner import run_speed


def main(argv: Sequence[str] | None = None) -> None:
    """Measure drafters' agreement with the fused group (GPU), or write the H9 report (CPU)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="measure agreement for a config")
    run.add_argument("config", type=Path, help="path to a speed config in configs/")
    report = commands.add_parser("report", help="write the H9 report of a finished run")
    report.add_argument("config", type=Path, help="the config the run used")
    report.add_argument("--out", type=Path, help="defaults to docs/results/<run name>")
    args = parser.parse_args(argv)
    config = load_speed_config(args.config)
    if args.command == "run":
        # Third-party libraries stay at WARNING; only this package reports progress.
        logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
        logging.getLogger("syncytium").setLevel(logging.INFO)
        run_speed(config, args.config)
    else:
        write_report(config, args.out or Path("docs/results") / config.name)


if __name__ == "__main__":
    main()
