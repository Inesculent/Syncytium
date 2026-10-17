"""Command line: ``holograph-report RESULTS_DIR [--out DIR]``."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from holograph.report.figures import all_figures
from holograph.report.tables import bf16_perplexity, load_results, summarize, to_markdown


def main(argv: Sequence[str] | None = None) -> None:
    """Write the summary table and figures for a sweep's results."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "results", type=Path, help="sweep output directory, e.g. results/pilot-0.8b"
    )
    parser.add_argument("--out", type=Path, help="defaults to docs/results/<sweep name>")
    args = parser.parse_args(argv)
    write_report(args.results, args.out or Path("docs/results") / args.results.name)


def write_report(results_dir: Path, out: Path) -> None:
    """Write ``summary.md``, ``summary.csv`` and each figure as SVG and PNG into ``out``."""
    results = load_results(results_dir / "results.jsonl")
    summary = summarize(results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(
        to_markdown(summary, bf16_perplexity(results)), encoding="utf-8"
    )
    summary.to_csv(out / "summary.csv", index=False)
    for name, figure in all_figures(results).items():
        figure.savefig(out / f"{name}.svg")
        figure.savefig(out / f"{name}.png", dpi=200)
