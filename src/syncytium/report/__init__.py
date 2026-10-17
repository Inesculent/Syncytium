"""Summary tables and paper figures from sweep results."""

from syncytium.report.cli import write_report
from syncytium.report.figures import all_figures
from syncytium.report.tables import bf16_perplexity, load_results, summarize, to_markdown

__all__ = [
    "all_figures",
    "bf16_perplexity",
    "load_results",
    "summarize",
    "to_markdown",
    "write_report",
]
