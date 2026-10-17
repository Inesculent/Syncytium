"""Model sizes compared at equal memory per node (``docs/method.md``, "Equal memory per node").

A node's memory can hold a large model at few bits or a smaller model at more bits. For each
staggered setting of each sweep, this compares, at the memory of one child:

- the best single model that fits in that memory: any measured model size, in bf16 or at any
  round-to-nearest bit width;
- the worst child, which is what one surviving node serves;
- the fleet: the worst rebuild from N - 1 children and the rebuild from all N.

Memory counts weights only (codes, scales and unquantized tensors), not the key/value cache or
runtime buffers. Parameter counts come from each model's config and are checked against the
memory every sweep logged, so a wrong count cannot go unnoticed.
"""

import json
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from syncytium.eval import ParameterCounts, child_footprint, dense_footprint
from syncytium.eval.memory import DENSE_BITS
from syncytium.models import parameter_counts
from syncytium.quant import Scheme
from syncytium.report.tables import SETTING, load_results, summarize

# Relative difference allowed between logged and recomputed bits per weight.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Sweep:
    """One sweep's results with the parameter counts of its model.

    Attributes:
        model: Model id or path the sweep evaluated.
        results: The sweep's results, from :func:`~syncytium.report.tables.load_results`.
        counts: Parameter counts of the model.
    """

    model: str
    results: pd.DataFrame
    counts: ParameterCounts

    @property
    def label(self) -> str:
        """Short model name, such as ``4B`` for ``Qwen/Qwen3.5-4B-Base``.

        Without a size in the name, it is the last part of the model's path.
        """
        name = self.model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        match = re.search(r"(\d+(?:\.\d+)?B)(?![a-z])", name)
        return match.group(1) if match else name

    def node_gb(self, bits: int, group_size: int) -> float:
        """Weight memory in GB of this model with ``bits``-bit codes, or bf16 for 16 bits."""
        if bits == DENSE_BITS:
            return dense_footprint(self.counts).gigabytes
        return child_footprint(self.counts, Scheme(bits, group_size, 1)).gigabytes


def load_sweep(directory: Path) -> Sweep:
    """Read a sweep's results and count its model's parameters from the model's config.

    Raises:
        ValueError: If the runs in ``directory`` evaluated different models, or the counted
            parameters do not reproduce the memory the sweep logged.
    """
    model = _model(directory / "runs.jsonl")
    sweep = Sweep(model, load_results(directory / "results.jsonl"), parameter_counts(model))
    check_counts(sweep)
    return sweep


def check_counts(sweep: Sweep) -> None:
    """Check the parameter counts against the bits per weight of every child and baseline.

    Raises:
        ValueError: On the first row whose logged memory the counts do not reproduce.
    """
    for row in sweep.results.to_dict("records"):
        if row["kind"] == "bf16":
            expected = dense_footprint(sweep.counts).bits_per_weight
        elif row["kind"] in ("child", "rtn"):
            scheme = Scheme(int(row["bits"]), int(row["group_size"]), 1)
            expected = child_footprint(sweep.counts, scheme).bits_per_weight
        else:
            continue
        if abs(row["bits_per_weight"] - expected) > TOLERANCE * expected:
            raise ValueError(
                f"{sweep.model}: counts give {expected} bits per weight for {row['id']}, "
                f"but the sweep logged {row['bits_per_weight']}"
            )


def single_models(sweeps: Sequence[Sweep]) -> pd.DataFrame:
    """Every single model the sweeps measured: bf16 and round-to-nearest at each bit width."""
    rows = []
    for sweep in sweeps:
        results = sweep.results
        for row in results[results["kind"].isin(("bf16", "rtn"))].to_dict("records"):
            dense = row["kind"] == "bf16"
            bits = DENSE_BITS if dense else int(row["bits"])
            group_size = 0 if dense else int(row["group_size"])
            name = "bf16" if dense else f"{bits}-bit"
            if not dense and row["rotation"] == "on":
                name += ", rotated"
            if not dense and row["scaling"] == "on":
                name += ", scaled"
            rows.append(
                {
                    "model": sweep.label,
                    "bits": bits,
                    "rotation": None if dense else row["rotation"],
                    "scaling": None if dense else row["scaling"],
                    "node_gb": sweep.node_gb(bits, group_size),
                    "perplexity": row["perplexity"],
                    "label": f"{sweep.label} {name}",
                }
            )
    # At equal memory the better model comes first, so frontier() skips the worse one.
    return pd.DataFrame(rows).sort_values(["node_gb", "perplexity"], ignore_index=True)


def frontier(singles: pd.DataFrame) -> pd.DataFrame:
    """The single models that beat every model needing less memory."""
    best = singles["perplexity"].cummin()
    improves = singles["perplexity"] < best.shift(fill_value=float("inf"))
    return singles[improves].reset_index(drop=True)


def equal_memory(sweeps: Sequence[Sweep]) -> pd.DataFrame:
    """One row per staggered setting: what one node, the fleet and the best single model do.

    Columns: ``model``, the setting, ``node_gb`` (one child), ``child_worst`` (one node
    alive), ``fleet_minus_one`` (worst rebuild from N - 1), ``fleet_all`` (rebuild from N),
    and ``best_single`` with its ``best_single_perplexity``: the best single model of any
    measured size that fits in ``node_gb``.
    """
    singles = single_models(sweeps)
    rows = []
    for sweep in sweeps:
        for row in summarize(sweep.results).to_dict("records"):
            node_gb = sweep.node_gb(int(row["bits"]), int(row["group_size"]))
            fits = singles[singles["node_gb"] <= node_gb]
            best = fits.loc[fits["perplexity"].idxmin()] if len(fits) else None
            rows.append(
                {
                    "model": sweep.label,
                    **{key: row[key] for key in SETTING},
                    "node_gb": node_gb,
                    "child_worst": row["child_worst"],
                    "fleet_minus_one": row["partial_worst"],
                    "fleet_all": row["rebuild"],
                    "best_single": None if best is None else best["label"],
                    "best_single_perplexity": None if best is None else best["perplexity"],
                }
            )
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, sweeps: Sequence[Sweep]) -> str:
    """The equal-memory table for N = 4 as a Markdown document, with how to read it first."""
    models = ", ".join(
        f"{s.label} ({s.counts.total / 1e9:.2f}B parameters, bf16 perplexity "
        f"{s.results.loc[s.results['kind'] == 'bf16', 'perplexity'].iloc[0]:.3f})"
        for s in sweeps
    )
    intro = (
        "Each row is one staggered setting with *N* = 4 children, compared at the weight "
        "memory of one child (`node_gb`; codes, scales and unquantized tensors, no key/value "
        "cache). `child_worst`: one node alive. `fleet_minus_one`: the worst rebuild from "
        "three. `fleet_all`: the rebuild from all four. `best_single`: the best single model "
        f"of any measured size that fits in the same memory. Models: {models}. The full table, "
        "for every *N*, is in `sizes.csv`.\n\n"
    )
    rows = table[table["num_children"] == 4].sort_values(
        ["node_gb", "phase", "rotation", "scaling"]
    )
    return intro + rows.round(3).to_markdown(index=False) + "\n"


def _model(runs: Path) -> str:
    with runs.open(encoding="utf-8") as file:
        configs = [tomllib.loads(json.loads(line)["config"]) for line in file if line.strip()]
    models = {config["model"]["path"] for config in configs}
    if len(models) != 1:
        raise ValueError(f"{runs} records {len(models)} models, expected one: {sorted(models)}")
    return models.pop()
