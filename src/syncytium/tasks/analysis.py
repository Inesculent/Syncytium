"""Pre-registered checks for a swarm task run (``docs/preregistration-swarm.md``).

H7 (fusion changes answers), H8 (a lone agent knows when to ask), H11 (graceful task-level
degradation) and H13 (surviving a member's loss mid-answer), all on GSM8K, computed from the
per-answer rows a run writes. Thresholds are the pre-registered ones and are not parameters.
"""

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

Items = Mapping[str, Sequence[Mapping[str, Any]]]

H7_RECOVERY, H7_POINTS, H7_ALPHA = 0.5, 0.03, 0.01
H8_FRACTION, H8_RECOVERY = 0.3, 0.5
H13_MARGIN, H13_Z = 0.02, 1.6448536269514722  # one-sided 95%


@dataclass(frozen=True)
class Check:
    """The outcome of one pre-registered test.

    Attributes:
        hypothesis: Its name, such as "H7".
        passed: Whether the pre-registered criterion holds.
        detail: The numbers behind the outcome, one line.
    """

    hypothesis: str
    passed: bool
    detail: str


def load_items(results: Path, task: str) -> dict[str, list[dict[str, Any]]]:
    """Per-answer rows of every condition of ``task``, keyed by condition id, in item order."""
    found = {}
    for path in sorted((results / "items" / task).glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        kind, _, rest = path.stem.partition("-")
        found[f"{kind}/{rest}" if rest else kind] = sorted(rows, key=lambda r: r["index"])
    return found


def accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of rows answered correctly."""
    return sum(r["correct"] for r in rows) / len(rows)


def mcnemar(first: Sequence[bool], second: Sequence[bool]) -> float:
    """Exact two-sided McNemar p-value for paired outcomes (a binomial test on discordances)."""
    gained = sum(a and not b for a, b in zip(first, second, strict=True))
    lost = sum(b and not a for a, b in zip(first, second, strict=True))
    n = gained + lost
    if n == 0:
        return 1.0
    tail = Fraction(sum(math.comb(n, i) for i in range(min(gained, lost) + 1)), 2**n)
    return min(1.0, float(2 * tail))


def escalate(
    entropy: Sequence[float], lone: Sequence[bool], fused: Sequence[bool], fraction: float
) -> float:
    """Accuracy when the ``fraction`` of answers with the highest entropy go to the fused group.

    Ties in entropy are broken by item order, so the result is deterministic.
    """
    count = math.ceil(fraction * len(entropy))
    order = sorted(range(len(entropy)), key=lambda i: (-entropy[i], i))
    escalated = set(order[:count])
    return sum(fused[i] if i in escalated else lone[i] for i in range(len(lone))) / len(lone)


def noninferiority_bound(differences: Sequence[float]) -> float:
    """One-sided 95% lower confidence bound of the mean of paired ``differences``."""
    spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
    return statistics.fmean(differences) - H13_Z * spread / math.sqrt(len(differences))


def children(items: Items) -> dict[int, Sequence[Mapping[str, Any]]]:
    """Rows of each lone child, keyed by child index."""
    return {_indices(c)[0]: rows for c, rows in items.items() if c.startswith("child/")}


def full_group(items: Items) -> str:
    """Id of the group of every child."""
    return f"group/k{','.join(map(str, sorted(children(items))))}"


def worst_child(items: Items) -> int:
    """Index of the least accurate child (the lowest index among ties)."""
    lone = children(items)
    return min(lone, key=lambda k: (accuracy(lone[k]), k))


def check_h7(items: Items) -> Check:
    """Four fused children recover ≥ 50% of the gap to bf16 and gain ≥ 3 points (p < 0.01)."""
    worst = children(items)[worst_child(items)]
    fused, bf16 = items[full_group(items)], items["bf16"]
    gap = accuracy(bf16) - accuracy(worst)
    points = accuracy(fused) - accuracy(worst)
    recovery = points / gap if gap > 0 else math.nan
    p = mcnemar(_correct(fused), _correct(worst))
    passed = gap > 0 and recovery >= H7_RECOVERY and points >= H7_POINTS and p < H7_ALPHA
    detail = (
        f"worst child {accuracy(worst):.3f}, fused {accuracy(fused):.3f}, "
        f"bf16 {accuracy(bf16):.3f}; gain {100 * points:+.1f} points, "
        f"{_percent(recovery)} of the gap, McNemar p = {p:.2g}"
    )
    return Check("H7", passed, detail)


def check_h8(items: Items) -> list[Check]:
    """Escalating the 30% most uncertain answers recovers ≥ 50% of the gain.

    The criterion is judged on the worst child, the one H7 uses; the other children are
    reported for reference.
    """
    fused = _correct(items[full_group(items)])
    worst = worst_child(items)
    checks = []
    for index, rows in sorted(children(items).items()):
        lone = _correct(rows)
        gain = statistics.fmean(fused) - statistics.fmean(lone)
        escalated = escalate([r["entropy"] for r in rows], lone, fused, H8_FRACTION)
        recovery = (escalated - statistics.fmean(lone)) / gain if gain > 0 else math.nan
        label = "H8" if index == worst else f"H8 (child {index}, reference)"
        detail = (
            f"child {index}: escalating 30% gives {escalated:.3f} "
            f"from {statistics.fmean(lone):.3f}, {_percent(recovery)} "
            f"of the {100 * gain:+.1f}-point gain (random: 30%)"
        )
        checks.append(Check(label, gain > 0 and recovery >= H8_RECOVERY, detail))
    return checks


def check_h11(items: Items) -> Check:
    """Mean accuracy does not fall with group size, and the worst N - 1 group beats every child."""
    by_size: dict[int, list[float]] = defaultdict(list)
    for condition, rows in items.items():
        if condition.startswith(("child/", "group/")):
            by_size[len(_indices(condition))].append(accuracy(rows))
    sizes = sorted(by_size)
    means = [statistics.fmean(by_size[s]) for s in sizes]
    rising = all(b >= a for a, b in pairwise(means))
    beats = min(by_size[sizes[-2]]) > max(by_size[1]) if len(sizes) > 2 else False
    described = ", ".join(f"{s}: {m:.3f}" for s, m in zip(sizes, means, strict=True))
    detail = (
        f"mean accuracy by group size {described}; worst group of {sizes[-2]} "
        f"{min(by_size[sizes[-2]]):.3f} vs best child {max(by_size[1]):.3f}"
    )
    return Check("H11", rising and beats, detail)


def check_h13(items: Items) -> Check | None:
    """Dropping a member halfway is non-inferior (2-point margin) to survivors from scratch."""
    drops = {_dropped(c): rows for c, rows in items.items() if c.startswith("drop/")}
    if not drops:
        return None
    everyone = sorted(children(items))
    per_item = []
    for i in range(len(items["bf16"])):
        differences = []
        for k, rows in drops.items():
            kept = [j for j in everyone if j != k]
            survivors = items[f"group/k{','.join(map(str, kept))}"]
            differences.append(rows[i]["correct"] - survivors[i]["correct"])
        per_item.append(statistics.fmean(differences))
    bound = noninferiority_bound(per_item)
    detail = (
        f"mean difference {100 * statistics.fmean(per_item):+.2f} points, one-sided 95% lower "
        f"bound {100 * bound:+.2f} (margin -2)"
    )
    return Check("H13", bound > -H13_MARGIN, detail)


def checks(items: Items) -> list[Check]:
    """Every pre-registered GSM8K check the items allow."""
    found = [check_h7(items), *check_h8(items), check_h11(items)]
    h13 = check_h13(items)
    return [*found, h13] if h13 else found


def report(results: Path) -> str:
    """Markdown summary of a run: accuracy of every condition and the hypothesis checks."""
    lines = [f"# {results.name}", ""]
    for task in ("gsm8k", "mmlu"):
        items = load_items(results, task)
        if not items:
            continue
        lines += [f"## {task.upper()}", "", "| Condition | Accuracy | Correct |", "|---|---:|---:|"]
        for condition, rows in items.items():
            correct = sum(r["correct"] for r in rows)
            lines.append(f"| {condition} | {accuracy(rows):.4f} | {correct}/{len(rows)} |")
        lines.append("")
        if task == "gsm8k" and "bf16" in items:
            lines += ["### Pre-registered checks", ""]
            lines += [
                f"- **{c.hypothesis}: {'pass' if c.passed else 'fail'}.** {c.detail}"
                for c in checks(items)
            ]
            lines.append("")
    return "\n".join(lines)


def write_report(results: Path, out: Path) -> None:
    """Write :func:`report` to ``out/summary.md``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(report(results), encoding="utf-8")


def _indices(condition: str) -> tuple[int, ...]:
    return tuple(int(k) for k in condition.partition("/k")[2].split(","))


def _dropped(condition: str) -> int:
    return int(condition.partition("/k")[2])


def _correct(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
    return [bool(r["correct"]) for r in rows]


def _percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100 * value:.0f}%"
