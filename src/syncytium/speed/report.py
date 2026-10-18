"""H9 from the traced rounds, the child's size and the measured agreement (no GPU needed).

The pre-registered pass point is a wired LAN and an edge-class device: 0.2 ms per step,
1 Gbit/s and 100 GB/s. Four fused devices pass if, with the best draft length, they reach at
least 25% of one solo device's tokens per second, drafting with the child that agrees with
the group least. The grid around the pass point is reported as well.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from syncytium.eval import child_footprint
from syncytium.models import parameter_counts
from syncytium.speed.config import SpeedConfig
from syncytium.speed.network import (
    Network,
    Swarm,
    fused_tokens_per_second,
    solo_seconds,
)
from syncytium.speed.rounds import model_rounds
from syncytium.speed.runner import ACCEPTANCE

PASS_POINT = Network(latency=0.2e-3, bandwidth=1e9, memory_bandwidth=100e9)
REQUIRED = 0.25
LATENCIES = (0.05e-3, 0.2e-3, 1e-3)
BANDWIDTHS = (1e9, 10e9)
MEMORY_BANDWIDTHS = (50e9, 100e9, 1000e9)


@dataclass(frozen=True)
class Outcome:
    """Fused against solo speed on one network.

    Attributes:
        network: The hardware.
        solo: Tokens per second of one device alone.
        fused: Tokens per second of the fused group with the best draft length.
        draft: That draft length (0 means no speculation was best).
    """

    network: Network
    solo: float
    fused: float
    draft: int

    @property
    def ratio(self) -> float:
        """Fused speed as a fraction of solo speed."""
        return self.fused / self.solo


def outcome(swarm: Swarm, network: Network, cycles: Mapping[int, float]) -> Outcome:
    """Best fused speed over no speculation and every measured draft length."""
    options = {0: fused_tokens_per_second(swarm, network)}
    options.update(
        {k: fused_tokens_per_second(swarm, network, k, tpc) for k, tpc in cycles.items()}
    )
    draft = max(options, key=lambda k: (options[k], -k))
    return Outcome(network, 1 / solo_seconds(swarm, network), options[draft], draft)


def load_acceptance(output: Path) -> list[dict[str, Any]]:
    """Rows written by :func:`~syncytium.speed.runner.measure_agreement`."""
    with (output / ACCEPTANCE).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def worst_drafter(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The child that agrees with the group least (the lowest index among ties)."""
    return min(rows, key=lambda r: (r["agreement"], r["child"]))


def report(config: SpeedConfig) -> str:
    """Markdown summary: rounds, agreement, the grid and the H9 check."""
    rounds = model_rounds(config.model.path)
    scheme = config.setting.scheme
    child = child_footprint(parameter_counts(config.model.path), scheme)
    swarm = Swarm(rounds, child.total_bits / 8, scheme.num_children)
    rows = load_acceptance(config.output)
    drafter = worst_drafter(rows)
    cycles = {int(k): v for k, v in drafter["tokens_per_cycle"].items()}
    per_token = sum(r.values for r in rounds)
    lines = [
        f"# {config.name}",
        "",
        f"{len(rounds)} rounds per fused token; each device contributes {per_token:,} values "
        f"per token ({per_token * 2 / 1e6:.2f} MB in bf16), of which {rounds[-1].values:,} "
        f"are output logits. One child is {child.gigabytes:.2f} GB.",
        "",
        "| Child | Agreement with the group | Tokens per cycle (k = 1, 4, 8) |",
        "|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda r: r["child"]):
        tpc = ", ".join(f"{r['tokens_per_cycle'][k]:.2f}" for k in ("1", "4", "8"))
        lines.append(f"| {r['child']} | {r['agreement']:.3f} | {tpc} |")
    lines += ["", _grid(swarm, cycles), "", _check(swarm, cycles, drafter["child"])]
    return "\n".join(lines) + "\n"


def write_report(config: SpeedConfig, out: Path) -> None:
    """Write :func:`report` to ``out/summary.md``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(report(config), encoding="utf-8")


def _grid(swarm: Swarm, cycles: Mapping[int, float]) -> str:
    lines = [
        "| Latency | Link | Memory | Solo tok/s | Fused tok/s | Draft | Fused / solo |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for latency, bandwidth, memory in product(LATENCIES, BANDWIDTHS, MEMORY_BANDWIDTHS):
        o = outcome(swarm, Network(latency, bandwidth, memory), cycles)
        lines.append(
            f"| {latency * 1e3:g} ms | {bandwidth / 1e9:g} Gbit/s | {memory / 1e9:g} GB/s "
            f"| {o.solo:.1f} | {o.fused:.1f} | {o.draft} | {o.ratio:.0%} |"
        )
    return "\n".join(lines)


def _check(swarm: Swarm, cycles: Mapping[int, float], drafter: int) -> str:
    o = outcome(swarm, PASS_POINT, cycles)
    verdict = "pass" if o.ratio >= REQUIRED else "fail"
    return (
        f"**H9: {verdict}.** At 0.2 ms, 1 Gbit/s and 100 GB/s, four fused devices reach "
        f"{o.fused:.1f} tokens/s against {o.solo:.1f} solo ({o.ratio:.0%}; at least 25% needed), "
        f"drafting {o.draft} tokens per cycle with child {drafter}."
    )
