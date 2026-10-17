"""Staggered children against simple baselines on the same nodes (``docs/method.md``).

Every scheme puts one shard on each of N nodes, and s of them survive an outage. For each s,
the comparison gives each scheme's perplexity in the worst case over which s survive (an
outage does not choose) and each node's weight memory:

- *staggered*: the children. The worst child for s = 1, the full rebuild for s = N, and the
  worst measured partial rebuild of s children otherwise (missing where none was measured).
- *replicated*: every node holds the same round-to-nearest model at the children's bit
  width. Same memory per node, and s does not matter.
- *coded* (k): every node holds the same base child plus one share of what the base lacks
  from the parent, erasure-coded so that any k shares recover it. Fewer than k survivors
  serve the base child; k or more rebuild the parent exactly. The parent's code differs from
  N times any child's code by one of N values, so the remainder takes log2 N bits per
  quantized weight and each share log2(N)/k. With k = 1 every node holds the whole parent,
  which is also where nested (Matryoshka-style) slicing of one stored model lands.

The base child is the one whose offset is closest to ½ (round-to-nearest when φ = 0), chosen
by rule rather than by measured perplexity.

Every number comes from conditions a sweep already measured; the baselines need no new runs.
"""

import math
from fractions import Fraction

import pandas as pd

from syncytium.report.tables import KINDS, SETTING

SCHEMES = ("staggered", "replicated", "coded")
LONG = [*SETTING, "scheme", "k", "survivors", "perplexity", "node_bpw"]
HALF = Fraction(1, 2)


def quantized_share(results: pd.DataFrame) -> float:
    """Fraction of all parameters that are quantized.

    Bits per weight grow by exactly this fraction per extra code bit, so it follows from the
    memory of two round-to-nearest conditions with the same group size.

    Raises:
        ValueError: If no group size has round-to-nearest results at two bit widths.
    """
    rtn = results[results["kind"] == "rtn"].drop_duplicates(["bits", "group_size"])
    for _, rows in rtn.groupby("group_size"):
        if len(rows) >= 2:
            low, high = rows.sort_values("bits").iloc[[0, -1]].itertuples()
            return (high.bits_per_weight - low.bits_per_weight) / (high.bits - low.bits)
    raise ValueError("needs round-to-nearest results at two bit widths with one group size")


def compare(results: pd.DataFrame) -> pd.DataFrame:
    """Every scheme at every survivor count, one row each, for every staggered setting.

    Columns: the setting, ``scheme``, ``k`` (coded schemes only), ``survivors``, worst-case
    ``perplexity`` and ``node_bpw`` (bits per weight on each node, scales and unquantized
    tensors included).
    """
    share = quantized_share(results)
    rtn = results[results["kind"] == "rtn"]
    replicated = rtn.groupby(["bits", "group_size", "rotation"])["perplexity"].first()
    staggered = results[results["kind"].isin(KINDS)]
    rows = []
    for setting, data in staggered.groupby(SETTING):
        bits, group_size, _, rotation, n = setting
        children = data[data["kind"] == "child"]
        node_bpw = float(children["bits_per_weight"].iloc[0])
        worst = data.groupby("present")["perplexity"].max()
        rebuild = float(data.loc[data["kind"] == "rebuild", "perplexity"].iloc[0])
        base = _base_child(children)
        same = replicated.get((bits, group_size, rotation), math.nan)
        for s in range(1, n + 1):
            rows.append([*setting, "staggered", None, s, worst.get(s, math.nan), node_bpw])
            rows.append([*setting, "replicated", None, s, same, node_bpw])
            for k in range(1, n + 1):
                value = rebuild if s >= k else base
                rows.append([*setting, "coded", k, s, value, node_bpw + share * math.log2(n) / k])
    frame = pd.DataFrame(rows, columns=LONG)
    frame["k"] = frame["k"].astype("Int64")
    return frame


def by_survivors(results: pd.DataFrame) -> pd.DataFrame:
    """One row per setting and survivor count: staggered against its two closest rivals.

    ``replicated`` costs the same memory per node; ``replicated_match_bpw`` is the extra
    memory per node at which replicated round-to-nearest becomes at least as good as
    staggered (missing if no measured bit width does). ``coded`` is the cheapest coded scheme
    that rebuilds the parent from this many survivors (k = s); ``coded_extra_bpw`` is what it
    costs each node on top of a child.
    """
    comparison = compare(results)
    keys = [*SETTING, "survivors"]
    cheapest = comparison["k"] == comparison["survivors"]
    columns = {
        "staggered": {"perplexity": "staggered", "node_bpw": "child_bpw"},
        "replicated": {"perplexity": "replicated"},
        "coded": {"perplexity": "coded", "node_bpw": "coded_bpw"},
    }
    parts = []
    for scheme, names in columns.items():
        rows = comparison[(comparison["scheme"] == scheme) & (cheapest | (scheme != "coded"))]
        parts.append(rows[[*keys, *names]].rename(columns=names))
    table = parts[0].merge(parts[1], on=keys).merge(parts[2], on=keys)
    table["coded_extra_bpw"] = table.pop("coded_bpw") - table["child_bpw"]
    rtn, share = results[results["kind"] == "rtn"], quantized_share(results)
    match = [_replicated_match(rtn, row, share) for row in table.to_dict("records")]
    table.insert(table.columns.get_loc("replicated") + 1, "replicated_match_bpw", match)
    return table.reset_index(drop=True)


def to_markdown(table: pd.DataFrame, share: float) -> str:
    """The survivor table as a Markdown document, with how to read it stated first."""
    intro = (
        "Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds "
        "the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is "
        "the extra memory per node at which it becomes at least as good as staggered. "
        "*coded*: every node holds one base child plus an erasure-coded share of the "
        "parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is "
        "its extra memory per node. "
        f"Quantized weights are {share:.1%} of all parameters. Blank staggered cells were "
        "not measured.\n\n"
    )
    return intro + table.round(3).to_markdown(index=False) + "\n"


def _replicated_match(rtn: pd.DataFrame, row: dict, share: float) -> float:
    """Extra bits per node until replicated round-to-nearest matches ``row["staggered"]``."""
    candidates = rtn[
        (rtn["group_size"] == row["group_size"])
        & (rtn["rotation"] == row["rotation"])
        & (rtn["bits"] >= row["bits"])
        & (rtn["perplexity"] <= row["staggered"])
    ]
    if candidates.empty:
        return math.nan
    return float((candidates["bits"].min() - row["bits"]) * share)


def _base_child(children: pd.DataFrame) -> float:
    """Perplexity of the child with offset closest to ½, the lowest index on ties."""
    rows = children.to_dict("records")
    best = min(rows, key=lambda r: (abs(Fraction(r["offsets"][0]) - HALF), r["indices"][0]))
    return float(best["perplexity"])
