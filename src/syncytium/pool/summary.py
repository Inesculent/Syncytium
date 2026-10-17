"""Summarizing a pooled-pipeline sweep against hypothesis H6 (``docs/hypothesis.md``).

Perplexity increases are compared in log space (nats per token above the full pool), where
the effects of losses on different stages would add if the stages degrade independently.
"""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

H6_MAX_GAP = 0.5  # any one lost device keeps perplexity within H3's bar of bf16
H6_MIN_R2 = 0.9  # stage-wise components predict the increase of random loss patterns
STATS = {"mean": "avg", "max": "worst"}


def load_pool_results(path: Path) -> pd.DataFrame:
    """Read a pooled sweep's ``results.jsonl``; ``estimator`` is "any" where it cannot matter."""
    with path.open(encoding="utf-8") as file:
        frame = pd.DataFrame([json.loads(line) for line in file if line.strip()])
    frame["estimator"] = frame["estimator"].fillna("any")
    frame["log_ppl"] = frame["perplexity"].map(math.log)
    frame["num_lost"] = frame["lost"].map(len)
    return frame


def additivity(results: pd.DataFrame) -> pd.DataFrame:
    """Measured and predicted log-perplexity increase of each random loss pattern.

    The prediction adds, over the stages that lost devices, the increase measured with only
    that stage in the same state.
    """
    full = _only(results, "full")
    n = int(full["num_children"])

    def label(estimator: str, alive: list[int]) -> str:
        return estimator if 1 < len(alive) < n else "any"

    components: dict[tuple[str, int, tuple[int, ...]], float] = {}
    for row in results[results["kind"].isin(["device", "stage", "component"])].itertuples():
        changed = [(s, alive) for s, alive in enumerate(row.alive) if len(alive) < n]
        if len(changed) == 1:
            stage, alive = changed[0]
            components[label(row.estimator, alive), stage, tuple(alive)] = (
                row.log_ppl - full["log_ppl"]
            )

    rows = []
    for row in results[results["kind"] == "random"].itertuples():
        predicted = sum(
            components[label(row.estimator, alive), s, tuple(alive)]
            for s, alive in enumerate(row.alive)
            if len(alive) < n
        )
        rows.append(
            {
                "id": row.id,
                "estimator": row.estimator,
                "num_lost": row.num_lost,
                "measured": row.log_ppl - full["log_ppl"],
                "predicted": predicted,
            }
        )
    return pd.DataFrame(rows, columns=["id", "estimator", "num_lost", "measured", "predicted"])


def per_estimator(frame: pd.DataFrame, estimators: list[str]) -> pd.DataFrame:
    """``frame`` with each "any" row repeated under every name in ``estimators``.

    An "any" row has the same weights under every estimator, so it belongs to each group.
    Without named estimators the frame is returned unchanged.
    """
    if not estimators:
        return frame
    shared = frame[frame["estimator"] == "any"]
    copies = [shared.assign(estimator=name) for name in estimators]
    return pd.concat([frame[frame["estimator"] != "any"], *copies], ignore_index=True)


def r_squared(measured: pd.Series, predicted: pd.Series) -> float:
    """Coefficient of determination of ``predicted`` for ``measured``; NaN without spread."""
    spread = ((measured - measured.mean()) ** 2).sum()
    if len(measured) < 2 or spread == 0:
        return math.nan
    return 1 - ((measured - predicted) ** 2).sum() / spread


def to_markdown(results: pd.DataFrame, layout: dict[str, Any] | None = None) -> str:
    """The pooled sweep as a Markdown document, with the H6 checks last."""
    bf16 = float(_only(results, "bf16")["perplexity"])
    full = float(_only(results, "full")["perplexity"])
    children = results.loc[results["kind"] == "child", "perplexity"]
    lines = [
        f"bf16 perplexity: **{bf16:.3f}** · full pool: **{full:.3f}** · "
        f"single child: **{children.mean():.3f}** (mean of {len(children)})",
        "",
    ]
    if layout:
        largest = max(layout["device_gb"])
        lines += [
            f"{layout['stages'] + 1} stages x {layout['children']} devices. The largest device "
            f"holds {largest:.3f} GB of weights; the bf16 model is {layout['dense_gb']:.3f} GB "
            f"({layout['dense_gb'] / largest:.1f}x larger).",
            "",
        ]

    estimators = sorted(set(results["estimator"]) - {"any"})
    devices = per_estimator(results[results["kind"] == "device"], estimators)
    by_stage = devices.assign(stage=devices["lost"].str[0].str[0]).pivot_table(
        index="stage", columns="estimator", values="perplexity", aggfunc=["mean", "max"]
    )
    by_stage.columns = [f"{estimator} ({STATS[stat]})" for stat, estimator in by_stage.columns]
    lines += ["## One device lost (perplexity by stage)", "", by_stage.round(3).to_markdown(), ""]

    stages = results[results["kind"] == "stage"].assign(stage=lambda f: f["lost"].str[0].str[0])
    single = stages.groupby("stage")["perplexity"].agg(["mean", "max"]).rename(columns=STATS)
    lines += ["## One stage on a single child", "", single.round(3).to_markdown(), ""]

    random = per_estimator(results[results["kind"] == "random"], estimators)
    losses = random.groupby(["estimator", "num_lost"])["perplexity"].agg(["count", "mean", "max"])
    lines += [
        "## Random device losses",
        "",
        losses.rename(columns=STATS).round(3).to_markdown(),
        "",
    ]

    fits = per_estimator(additivity(results), estimators)
    fit_rows = [
        {
            "estimator": estimator,
            "patterns": len(group),
            "r_squared": r_squared(group["measured"], group["predicted"]),
            "measured / predicted": group["measured"].sum() / group["predicted"].sum(),
        }
        for estimator, group in fits.groupby("estimator")
    ]
    fit_table = pd.DataFrame(fit_rows).round(3).to_markdown(index=False) if fit_rows else "none"
    lines += ["## Additivity across stages (log perplexity)", "", fit_table, ""]

    worst_gap = devices["perplexity"].max() - bf16
    lines += [
        "## Hypothesis checks",
        "",
        f"- H6b, one lost device stays within {H6_MAX_GAP} of bf16: worst gap "
        f"{worst_gap:.3f}, **{'pass' if worst_gap <= H6_MAX_GAP else 'fail'}**",
    ]
    for row in fit_rows:
        verdict = "pass" if row["r_squared"] >= H6_MIN_R2 else "fail"
        lines.append(
            f"- H6c, stage-wise prediction R² ≥ {H6_MIN_R2} ({row['estimator']}): "
            f"{row['r_squared']:.3f}, **{verdict}**"
        )
    return "\n".join(lines) + "\n"


def write_pool_report(results_dir: Path, out: Path) -> None:
    """Write ``summary.md`` and ``additivity.csv`` for a pooled sweep into ``out``."""
    results = load_pool_results(results_dir / "results.jsonl")
    layout_path = results_dir / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(to_markdown(results, layout), encoding="utf-8")
    additivity(results).to_csv(out / "additivity.csv", index=False)


def _only(results: pd.DataFrame, kind: str) -> pd.Series:
    return results[results["kind"] == kind].iloc[0]
