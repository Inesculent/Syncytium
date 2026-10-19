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
from matplotlib.ticker import FuncFormatter, LogFormatter, NullFormatter

from syncytium.report.baselines import compare, quantized_share
from syncytium.report.sizes import Sweep, equal_memory, frontier, single_models
from syncytium.report.tables import KINDS, bf16_perplexity

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dddcd7"
PANEL_SIZE = (3.2, 3.0)
MEMORY_TICKS = (0.25, 0.5, 1, 2, 4, 8, 16)
PERPLEXITY_TICKS = (5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100)
# Children per set in the baseline figures; the coded baseline rebuilds from any N - 1 nodes.
BASELINE_CHILDREN = 4


def all_figures(results: pd.DataFrame) -> dict[str, Figure]:
    """Every figure for a results frame, keyed by file name (without extension)."""
    figures = {}
    # The offsets figure shows unscaled children only; a run of scaled settings has none.
    if ((results["kind"] == "child") & (results["scaling"] == "off")).any():
        figures["offsets"] = offsets_figure(results)
    baselines = has_baselines(results)
    for variant in _settings(results):
        suffix = _suffix(*variant)
        figures[f"recovery-{suffix}"] = recovery_figure(results, *variant)
        figures[f"pareto-{suffix}"] = pareto_figure(results, *variant)
        if baselines:
            figures[f"baselines-{suffix}"] = baseline_figure(results, *variant)
    return figures


def sizes_figures(sweeps: list[Sweep]) -> dict[str, Figure]:
    """The equal-memory figure for each phase, rotation and scaling the sweeps share."""
    shared = set.intersection(*(set(_settings(s.results)) for s in sweeps))
    return {
        f"sizes-{_suffix(*variant)}": sizes_figure(sweeps, *variant) for variant in sorted(shared)
    }


def sizes_figure(sweeps: list[Sweep], phase: float, rotation: str, scaling: str = "off") -> Figure:
    """Perplexity against weight memory per node, across model sizes.

    The step line is the best single model that fits in each amount of memory. Each 3- or
    4-bit staggered setting with N = ``BASELINE_CHILDREN`` adds two points at the memory of
    one child: its worst child (one node alive) and its worst rebuild from N - 1 children.
    2-bit children are left out; they are unusable at every measured size.
    """
    n = BASELINE_CHILDREN
    best = frontier(single_models(sweeps))
    table = equal_memory(sweeps)
    table = table[
        (table["num_children"] == n)
        & (table["phase"] == phase)
        & (table["rotation"] == rotation)
        & (table["scaling"] == scaling)
        & (table["bits"] >= 3)
    ]
    figure, (axis,) = _panels(1)
    figure.set_size_inches(PANEL_SIZE[0] * 2, PANEL_SIZE[1] * 1.4)
    axis.set_xscale("log")
    axis.step(
        best["node_gb"],
        best["perplexity"],
        where="post",
        **_mark(0, "-"),
        label="best single model",
    )
    axis.plot(table["node_gb"], table["child_worst"], **_mark(1, "none"), label="one node alive")
    label = f"any {n - 1} of {n} alive"
    axis.plot(table["node_gb"], table["fleet_minus_one"], **_mark(2, "none"), label=label)
    for row in table.to_dict("records"):
        # 3- and 4-bit settings of one model sit close together; label them on either side.
        left = row["bits"] == 3
        axis.annotate(
            f"{row['model']} {row['bits']}-bit",
            (row["node_gb"], row["fleet_minus_one"]),
            xytext=(-4 if left else 4, -10),
            textcoords="offset points",
            ha="right" if left else "left",
            color=TEXT,
            fontsize=7,
        )
    # Leave out the unusable 2-bit single models, which would flatten everything else.
    shown = [*table["child_worst"], *table["fleet_minus_one"]]
    axis.set_ylim(min(*shown, *best["perplexity"]) * 0.95, max(shown) * 1.15)
    for ticks, scale, (low, high) in (
        (MEMORY_TICKS, axis.xaxis, axis.get_xlim()),
        (PERPLEXITY_TICKS, axis.yaxis, axis.get_ylim()),
    ):
        inside = [tick for tick in ticks if low <= tick <= high]
        if inside:  # otherwise keep matplotlib's own ticks
            scale.set_ticks(inside)
            scale.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            scale.set_minor_formatter(NullFormatter())
    axis.set_xlabel("weight memory per node (GB, log scale)", color=TEXT)
    axis.set_ylabel("WikiText-2 perplexity (log scale)", color=TEXT)
    axis.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    return figure


def has_baselines(results: pd.DataFrame) -> bool:
    """Whether ``results`` support the baseline comparison at ``BASELINE_CHILDREN`` children."""
    try:
        quantized_share(results)
    except ValueError:
        return False
    return bool((results["num_children"] == BASELINE_CHILDREN).any())


def baseline_figure(
    results: pd.DataFrame, phase: float, rotation: str, scaling: str = "off"
) -> Figure:
    """Worst-case perplexity against surviving nodes: staggered children and two baselines.

    One panel per bit width, for N = ``BASELINE_CHILDREN``. The coded baseline rebuilds the
    parent from any N - 1 nodes, at the extra memory per node given in its label.
    """
    n, k = BASELINE_CHILDREN, BASELINE_CHILDREN - 1
    data = compare(results)
    data = data[
        (data["phase"] == phase)
        & (data["rotation"] == rotation)
        & (data["scaling"] == scaling)
        & (data["num_children"] == n)
    ]
    panels = sorted(data["bits"].unique())
    figure, axes = _panels(len(panels))
    for axis, bits in zip(axes, panels, strict=True):
        panel = data[data["bits"] == bits]
        staggered = panel[panel["scheme"] == "staggered"].dropna(subset=["perplexity"])
        replicated = panel[panel["scheme"] == "replicated"]
        coded = panel[(panel["scheme"] == "coded") & (panel["k"] == k)]
        extra = coded["node_bpw"].iloc[0] - replicated["node_bpw"].iloc[0]
        _bridged(axis, staggered.set_index("survivors")["perplexity"], slot=0, label="staggered")
        axis.plot(replicated["survivors"], replicated["perplexity"], **_mark(1), label="same model")
        label = f"coded, any {k} (+{extra:.2f} bits)"
        axis.plot(coded["survivors"], coded["perplexity"], **_mark(2), label=label)
        _reference(axis, bf16_perplexity(results))
        axis.set_xticks(range(1, n + 1))
        axis.set_xlabel(f"nodes alive (of {n})", color=TEXT)
        axis.set_title(f"{bits}-bit children", color=TEXT, fontsize=10)
    axes[0].set_ylabel("worst-case perplexity (log scale)", color=TEXT)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
    )
    return figure


def recovery_figure(
    results: pd.DataFrame, phase: float, rotation: str, scaling: str = "off"
) -> Figure:
    """Perplexity against the number of children present, one panel per bit width."""
    data = _select(results, phase, rotation, scaling)
    data = data[data["kind"].isin(KINDS)]
    panels = sorted(data["bits"].unique())
    figure, axes = _panels(len(panels))
    for axis, bits in zip(axes, panels, strict=True):
        _recovery_panel(axis, data[data["bits"] == bits], bf16_perplexity(results))
        axis.set_title(f"{bits}-bit children", color=TEXT, fontsize=10)
    axes[0].set_ylabel("WikiText-2 perplexity (log scale)", color=TEXT)
    return figure


def pareto_figure(
    results: pd.DataFrame, phase: float, rotation: str, scaling: str = "off"
) -> Figure:
    """Perplexity against bits per weight: round-to-nearest, single children and rebuilds."""
    data = _select(results, phase, rotation, scaling)
    rtn = _select(results[results["kind"] == "rtn"], None, rotation, scaling)
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
    """Perplexity of each child against its grid offset, one panel per bit width.

    Only unscaled children are shown, so each curve is one rotation setting.
    """
    children = results[(results["kind"] == "child") & (results["scaling"] == "off")].copy()
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
        _bridged(axis, curve, slot)
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


def _bridged(axis: Axes, curve: pd.Series, slot: int, label: str | None = None) -> None:
    """Plot ``curve`` (indexed by a count), dotted across counts that were not measured.

    Sweeps measure one child and N - max_missing ... N children, so the counts in between
    are often missing.
    """
    for start, end in zip(curve.index[:-1], curve.index[1:], strict=True):
        segment = curve.loc[[start, end]]
        linestyle = "-" if end - start == 1 else ":"
        axis.plot(segment.index, segment.values, **_mark(slot, linestyle))
    # One invisible handle carries the legend entry, so segments are not listed separately.
    axis.plot([], [], **_mark(slot), label=label)


def _select(
    results: pd.DataFrame, phase: float | None, rotation: str, scaling: str
) -> pd.DataFrame:
    """Rows of one variant; ``phase`` ``None`` keeps every phase."""
    chosen = (results["rotation"] == rotation) & (results["scaling"] == scaling)
    if phase is not None:
        chosen &= results["phase"] == phase
    return results[chosen]


def _suffix(phase: float, rotation: str, scaling: str) -> str:
    """File name suffix of a variant; unscaled names predate scaling and stay unchanged."""
    return f"p{phase:g}-rot-{rotation}" + ("-scaled" if scaling == "on" else "")


def _settings(results: pd.DataFrame) -> list[tuple[float, str]]:
    staggered = results[results["kind"].isin(KINDS)]
    columns = (staggered["phase"], staggered["rotation"], staggered["scaling"])
    return sorted(set(zip(*columns, strict=True)))


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
    # Plain numbers (14.5, not 1.45e1); minor ticks are labeled only on short ranges.
    axis.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    axis.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.5)))
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.tick_params(which="both", colors=MUTED, labelsize=8)
