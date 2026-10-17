"""Command lines for reports.

``syncytium-report RESULTS_DIR [--out DIR]`` writes one sweep's tables and figures;
``syncytium-compare RESULTS_DIR... [--out DIR]`` compares model sizes at equal memory per node.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from syncytium.report import baselines, sizes
from syncytium.report.figures import all_figures, has_baselines, sizes_figures
from syncytium.report.tables import bf16_perplexity, load_results, summarize, to_markdown


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
    """Write ``summary.md``, ``summary.csv`` and each figure as SVG and PNG into ``out``.

    With round-to-nearest baselines in the log, also write ``baselines.md`` (staggered
    against the baselines by surviving nodes) and ``baselines.csv`` (every scheme).
    """
    results = load_results(results_dir / "results.jsonl")
    summary = summarize(results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(
        to_markdown(summary, bf16_perplexity(results)), encoding="utf-8"
    )
    summary.to_csv(out / "summary.csv", index=False)
    if has_baselines(results):
        table, share = baselines.by_survivors(results), baselines.quantized_share(results)
        (out / "baselines.md").write_text(baselines.to_markdown(table, share), encoding="utf-8")
        baselines.compare(results).to_csv(out / "baselines.csv", index=False)
    for name, figure in all_figures(results).items():
        figure.savefig(out / f"{name}.svg")
        figure.savefig(out / f"{name}.png", dpi=200)


def compare_main(argv: Sequence[str] | None = None) -> None:
    """Compare the model sizes of several sweeps at equal memory per node."""
    parser = argparse.ArgumentParser(description=compare_main.__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="sweep output directories")
    parser.add_argument("--out", type=Path, default=Path("docs/results/sizes"))
    args = parser.parse_args(argv)
    write_comparison(args.results, args.out)


def write_comparison(results_dirs: Sequence[Path], out: Path) -> None:
    """Write ``sizes.md``, ``sizes.csv`` and the equal-memory figures into ``out``."""
    sweeps = [sizes.load_sweep(directory) for directory in results_dirs]
    table = sizes.equal_memory(sweeps)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sizes.md").write_text(sizes.to_markdown(table, sweeps), encoding="utf-8")
    table.to_csv(out / "sizes.csv", index=False)
    for name, figure in sizes_figures(sweeps).items():
        figure.savefig(out / f"{name}.svg")
        figure.savefig(out / f"{name}.png", dpi=200)
