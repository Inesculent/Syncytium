"""Turning a sweep's results log into a summary table per setting.

Each row of the summary is one setting (bits, group size, phase, rotation, N) and answers
the hypotheses directly: whether every child is usable alone (H2), how close the full rebuild
comes to bf16 (H3) and whether losing one child still beats any single child (H4). The
thresholds are the ones fixed in docs/hypothesis.md after the 0.8B pilot.
"""

import json
from pathlib import Path

import pandas as pd

SETTING = ["bits", "group_size", "phase", "rotation", "num_children"]
KINDS = ("child", "rebuild", "partial")
H2_MAX_RATIO = 1.5
H3_MAX_GAP = 0.5


def load_results(path: Path) -> pd.DataFrame:
    """Read ``results.jsonl`` into a frame with a ``rotation`` column ("on" or "off")."""
    with path.open(encoding="utf-8") as file:
        frame = pd.DataFrame([json.loads(line) for line in file if line.strip()])
    # Rows logged before rotation existed have no rotation_seed field.
    seeds = frame.get("rotation_seed", pd.Series(index=frame.index, dtype=object))
    frame["rotation"] = seeds.notna().map({True: "on", False: "off"})
    # The bf16 row has no scheme; nullable integers keep these columns integral.
    integers = ["bits", "group_size", "num_children"]
    frame[integers] = frame[integers].astype("Int64")
    frame["present"] = frame["indices"].map(len)
    return frame


def bf16_perplexity(results: pd.DataFrame) -> float:
    """Perplexity of the unquantized model."""
    return float(results.loc[results["kind"] == "bf16", "perplexity"].iloc[0])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per setting with child, rebuild and partial-rebuild statistics.

    Partial-rebuild statistics cover rebuilds missing exactly one child, the case H4 judges,
    even when the log also holds rebuilds with more children missing.
    """
    frames = {kind: results[results["kind"] == kind] for kind in KINDS}
    partial = frames["partial"]
    frames["partial"] = partial[partial["present"] == partial["num_children"] - 1]
    by_kind = {kind: frame.groupby(SETTING) for kind, frame in frames.items()}
    summary = pd.concat(
        [
            by_kind["child"]["perplexity"].agg(
                child_best="min", child_mean="mean", child_worst="max"
            ),
            by_kind["rebuild"]["perplexity"].first().rename("rebuild"),
            by_kind["partial"]["perplexity"].agg(partial_mean="mean", partial_worst="max"),
            by_kind["child"]["bits_per_weight"].first().rename("child_bpw"),
            by_kind["rebuild"]["bits_per_weight"].first().rename("parent_bpw"),
        ],
        axis=1,
    ).reset_index()
    return _add_hypothesis_checks(summary, bf16_perplexity(results))


def _add_hypothesis_checks(summary: pd.DataFrame, bf16: float) -> pd.DataFrame:
    # An outage does not choose which children survive, so H2 judges the worst one.
    summary["child_ratio"] = summary["child_worst"] / bf16
    summary["rebuild_gap"] = summary["rebuild"] - bf16
    summary["h2"] = summary["child_ratio"] <= H2_MAX_RATIO
    summary["h3"] = summary["rebuild_gap"] <= H3_MAX_GAP
    # H4 needs a partial rebuild of at least two children; with N = 2 the "partial" is a
    # single child, so those settings get <NA> rather than False.
    beats_best_child = (summary["partial_worst"] < summary["child_best"]).astype("boolean")
    testable = summary["partial_worst"].notna() & (summary["num_children"] > 2)
    summary["h4"] = beats_best_child.where(testable)
    return summary


def to_markdown(summary: pd.DataFrame, bf16: float) -> str:
    """Summary as a Markdown document with the bf16 reference stated first."""
    table = summary.round(3).to_markdown(index=False)
    return f"bf16 perplexity: **{bf16:.3f}**\n\n{table}\n"
