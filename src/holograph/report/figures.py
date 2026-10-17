"""Paper figures drawn from a sweep's results log.

Conventions: at most three series per chart, taken in order from the first three slots of
the validated categorical palette; a single y-axis; thin lines; marker shapes and direct
labels as secondary encoding (slot 3 is below 3:1 contrast); text in text colors, never in
series colors. The summary table is the accessible table view of every figure.
"""

from fractions import Fraction

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from holograph.report.tables import KINDS, bf16_perplexity

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dddcd7"
PANEL_SIZE = (3.2, 3.0)


def all_figures(results: pd.DataFrame) -> dict[str, Figure]:
    """Every figure for a results frame, keyed by file name (without extension)."""
    figures = {"offsets": offsets_figure(results)}
    for phase, rotation in _settings(results):
        suffix = f"p{phase:g}-rot-{rotation}"
        figures[f"recovery-{suffix}"] = recovery_figure(results, phase, rotation)
        figures[f"pareto-{suffix}"] = pareto_figure(results, phase, rotation)
    return figures


def recovery_figure(results: pd.DataFrame, phase: float, rotation: str) -> Figure:
    """Perplexity against the number of children present, one panel per bit width."""
    data = _select(results, phase, rotation)
    data = data[data["kind"].isin(KINDS)]
    panels = sorted(data["bits"].unique())
    figure, axes = _panels(len(panels))
    for axis, bits in zip(axes, panels, strict=True):
        _recovery_panel(axis, data[data["bits"] == bits], bf16_perplexity(results))
        axis.set_title(f"{bits}-bit children", color=TEXT, fontsize=10)
    axes[0].set_ylabel("WikiText-2 perplexity (log scale)", color=TEXT)
    return figure


def pareto_figure(results: pd.DataFrame, phase: float, rotation: str) -> Figure:
    """Perplexity against bits per weight: round-to-nearest, single children and rebuilds."""
    data = _select(results, phase, rotation)
    rtn = results[(results["kind"] == "rtn") & (results["rotation"] == rotation)]
    figure, (axis,) = _panels(1)
    series = [
        ("round-to-nearest", rtn.sort_values("bits_per_weight"), "-"),
        ("single child", data[data["kind"] == "child"], "none"),
        ("rebuild from all N", data[data["kind"] == "rebuild"], "none"),
    ]
    for slot, (label, rows, line) in enumerate(series):
        axis.plot(rows["bits_per_weight"], rows["perplexity"], **_mark(slot, line), label=label)
    _reference(axis, bf16_perplexity(results))
    axis.set_xlabel("bits per weight", color=TEXT)
    axis.set_ylabel("WikiText-2 perplexity (log scale)", color=TEXT)
    axis.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    return figure


def offsets_figure(results: pd.DataFrame) -> Figure:
    """Perplexity of each child against its grid offset, one panel per bit width."""
    children = results[results["kind"] == "child"].copy()
    children["offset"] = children["offsets"].map(lambda o: float(Fraction(o[0])))
    panels = sorted(children["bits"].unique())
    figure, axes = _panels(len(panels))
    for axis, bits in zip(axes, panels, strict=True):
        panel = children[children["bits"] == bits]
        for slot, (rotation, rows) in enumerate(panel.groupby("rotation")):
            curve = rows.groupby("offset")["perplexity"].mean()
            axis.plot(curve.index, curve.values, **_mark(slot), label=f"rotation {rotation}")
        _style(axis)
        axis.set_title(f"{bits}-bit children", color=TEXT, fontsize=10)
        axis.set_xlabel("grid offset (units of Δ)", color=TEXT)
    axes[0].set_ylabel("WikiText-2 perplexity (log scale)", color=TEXT)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    return figure


def _recovery_panel(axis: Axes, data: pd.DataFrame, bf16: float) -> None:
    for slot, (n, rows) in enumerate(data.groupby("num_children")):
        curve = rows.groupby("present")["perplexity"].mean()
        axis.plot(curve.index, curve.values, **_mark(slot), label=f"N = {n}")
        axis.annotate(
            f"N = {n}",
            (curve.index[-1], curve.values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            color=TEXT,
            fontsize=8,
            va="center",
        )
    _reference(axis, bf16)
    axis.set_xlabel("children present", color=TEXT)


def _select(results: pd.DataFrame, phase: float, rotation: str) -> pd.DataFrame:
    return results[(results["phase"] == phase) & (results["rotation"] == rotation)]


def _settings(results: pd.DataFrame) -> list[tuple[float, str]]:
    staggered = results[results["kind"].isin(KINDS)]
    return sorted(set(zip(staggered["phase"], staggered["rotation"], strict=True)))


def _panels(count: int) -> tuple[Figure, list[Axes]]:
    figure = Figure(figsize=(PANEL_SIZE[0] * count, PANEL_SIZE[1]), layout="constrained")
    axes = list(figure.subplots(1, count, squeeze=False)[0])
    for axis in axes:
        _style(axis)
    return figure, axes


def _mark(slot: int, linestyle: str = "-") -> dict:
    """Line and marker style of categorical slot ``slot``."""
    if slot >= len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series per chart; facet or filter the data")
    return {
        "color": SERIES[slot],
        "marker": MARKERS[slot],
        "linestyle": linestyle,
        "linewidth": 2,
        "markersize": 6,
    }


def _reference(axis: Axes, bf16: float) -> None:
    axis.axhline(bf16, color=MUTED, linewidth=1, linestyle="--")
    axis.annotate(
        "bf16",
        (0, bf16),
        xycoords=("axes fraction", "data"),
        xytext=(2, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )


def _style(axis: Axes) -> None:
    axis.set_yscale("log")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.tick_params(which="both", colors=MUTED, labelsize=8)
