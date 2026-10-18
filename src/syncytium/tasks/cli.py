"""Command line: ``syncytium-tasks run CONFIG`` and ``syncytium-tasks report RESULTS_DIR``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from syncytium.tasks.analysis import write_report
from syncytium.tasks.config import load_task_config
from syncytium.tasks.runner import run_tasks


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate the swarm conditions on GSM8K and MMLU, or summarize a finished run."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate the conditions in a config")
    run.add_argument("config", type=Path, help="path to a task config in configs/")
    report = commands.add_parser("report", help="write the summary of a finished run")
    report.add_argument("results", type=Path, help="run output directory")
    report.add_argument("--out", type=Path, help="defaults to docs/results/<run name>")
    args = parser.parse_args(argv)
    if args.command == "run":
        # Third-party libraries stay at WARNING; only this package reports progress.
        logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
        logging.getLogger("syncytium").setLevel(logging.INFO)
        run_tasks(load_task_config(args.config), args.config)
    else:
        write_report(args.results, args.out or Path("docs/results") / args.results.name)


if __name__ == "__main__":
    main()
