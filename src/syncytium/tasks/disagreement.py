"""Escalation on disagreement (``docs/preregistration-disagreement.md``, H8b).

The requester (the least accurate lone child) and a partner answer alone. When their final
numbers differ, or either has none, the problem goes to a stronger model (the full group, or
for a static split the 6-bit model); otherwise the requester's answer stands. The threshold
values are the pre-registered ones and are not parameters.
"""

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from syncytium.tasks.analysis import Check, Items, children, full_group, worst_child
from syncytium.tasks.gsm8k import is_correct

H8B_ESCALATED, H8B_RECOVERY, H8B_POINTS = 0.5, 0.75, 0.03

Rows = Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class Policy:
    """The outcome of escalating on one pair's disagreements.

    Attributes:
        partner: The partner child.
        escalated: Fraction of problems escalated.
        accuracy: Accuracy when escalated problems take the stronger model's answers.
    """

    partner: int
    escalated: float
    accuracy: float


def agree(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Whether two answers end in the same number; an answer without one never agrees."""
    if first["prediction"] is None or second["prediction"] is None:
        return False
    return is_correct(first["prediction"], second["prediction"])


def policy(requester: Rows, partner: Rows, stronger: Rows, index: int) -> Policy:
    """Escalate to ``stronger`` wherever ``requester`` and ``partner`` disagree."""
    escalate = [not agree(r, p) for r, p in zip(requester, partner, strict=True)]
    correct = [
        s["correct"] if e else r["correct"]
        for e, r, s in zip(escalate, requester, stronger, strict=True)
    ]
    return Policy(index, statistics.fmean(escalate), statistics.fmean(correct))


def policies(items: Items, stronger: str) -> list[Policy]:
    """The worst child asking, with each other child as partner, escalating to ``stronger``."""
    lone = children(items)
    requester = worst_child(items)
    return [
        policy(lone[requester], lone[k], items[stronger], k) for k in sorted(lone) if k != requester
    ]


def check_h8b(items: Items) -> list[Check]:
    """Every partner escalates at most 50% and recovers at least 75% of a 3-point gain.

    Returns the overall verdict first, then one line per partner.
    """
    requester = worst_child(items)
    base = statistics.fmean(r["correct"] for r in children(items)[requester])
    gain = statistics.fmean(r["correct"] for r in items[full_group(items)]) - base
    lines = []
    for p in policies(items, full_group(items)):
        recovery = (p.accuracy - base) / gain if gain > 0 else float("nan")
        passed = gain >= H8B_POINTS and p.escalated <= H8B_ESCALATED and recovery >= H8B_RECOVERY
        detail = (
            f"child {requester} asking, child {p.partner} as partner: escalates "
            f"{p.escalated:.0%}, accuracy {p.accuracy:.3f}, {recovery:.0%} of the gain "
            f"(random escalation at the same rate: {p.escalated:.0%})"
        )
        lines.append(Check(f"H8b, partner {p.partner}", passed, detail))
    overall = (
        f"gain of the full group over child {requester}: {100 * gain:+.1f} points; "
        f"{sum(c.passed for c in lines)} of {len(lines)} partners pass"
    )
    return [Check("H8b", all(c.passed for c in lines), overall), *lines]
