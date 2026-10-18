"""The swarm against static splits of the same devices (H10b), and the confirmation report.

Both hypotheses are from ``docs/preregistration-disagreement.md``; H8b's check is in
:mod:`syncytium.tasks.disagreement`.

Throughput is problems per second per answer token on four devices kept busy, from the
device-seconds each problem needs per answer token. A partition dedicated to one role uses only
its own devices, so a static split runs at the pace of whichever partition saturates first.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig

from syncytium.eval import child_footprint
from syncytium.models import parameter_counts
from syncytium.quant import Scheme
from syncytium.speed.config import SpeedConfig
from syncytium.speed.network import Network, Swarm
from syncytium.speed.report import PASS_POINT, load_acceptance, outcome, worst_drafter
from syncytium.speed.rounds import model_rounds
from syncytium.tasks.analysis import Check, Items, accuracy, children, full_group, load_items
from syncytium.tasks.disagreement import check_h8b, policies

DEVICES = 4
LARGE_BITS = 6  # the static splits' large model


@dataclass(frozen=True)
class Costs:
    """Seconds per answer token of each kind of answer on the pass-point network.

    Attributes:
        solo: One lone child on one device.
        fused: The full group on all four devices (its best draft length).
        large: One read of the large model's weights, at one device's memory bandwidth.
        hop: Passing one bf16 hidden vector to the next pipeline stage.
    """

    solo: float
    fused: float
    large: float
    hop: float

    def pipeline(self, devices: int) -> float:
        """Device-seconds per token of the large model pipelined over ``devices``."""
        return self.large + (devices - 1) * self.hop


@dataclass(frozen=True)
class System:
    """One way to use the four devices.

    Attributes:
        name: What it is.
        accuracy: GSM8K accuracy.
        throughput: Problems per second per answer token.
    """

    name: str
    accuracy: float
    throughput: float


def costs(config: SpeedConfig, network: Network = PASS_POINT) -> Costs:
    """Per-token costs at the model size of ``config``, from H9's model and measurements."""
    scheme = config.setting.scheme
    counts = parameter_counts(config.model.path)
    child = child_footprint(counts, scheme).total_bits / 8
    large_scheme = Scheme(
        LARGE_BITS, scheme.group_size, 1, 0.5, scheme.rotation_seed, scheme.scaling
    )
    large = child_footprint(counts, large_scheme).total_bits / 8
    swarm = Swarm(model_rounds(config.model.path), child, scheme.num_children)
    drafter = worst_drafter(load_acceptance(config.output))
    cycles = {int(k): v for k, v in drafter["tokens_per_cycle"].items()}
    hidden = AutoConfig.from_pretrained(config.model.path).get_text_config().hidden_size
    return Costs(
        solo=child / network.memory_bandwidth,
        fused=1 / outcome(swarm, network, cycles).fused,
        large=large / network.memory_bandwidth,
        hop=network.latency + hidden * 2 * 8 / network.bandwidth,
    )


def systems(items: Items, cost: Costs) -> list[System]:
    """The swarm, both static splits and the floor, averaged over the requester's partners."""
    large = f"rtn/b{LARGE_BITS}"
    fused = policies(items, full_group(items))
    split = policies(items, large)
    escalated = statistics.fmean(p.escalated for p in fused)
    two_answers = 2 * cost.solo
    swarm = DEVICES / (two_answers + escalated * DEVICES * cost.fused)
    agents = 2 / two_answers
    large_side = 2 / (escalated * cost.pipeline(2)) if escalated else float("inf")
    return [
        System("swarm", statistics.fmean(p.accuracy for p in fused), swarm),
        System(
            "split A: two agents + 6-bit on two devices",
            statistics.fmean(p.accuracy for p in split),
            min(agents, large_side),
        ),
        System(
            "split B: 6-bit on four devices", accuracy(items[large]), DEVICES / cost.pipeline(4)
        ),
        System(
            "floor: four lone agents",
            statistics.fmean(accuracy(rows) for rows in children(items).values()),
            DEVICES / cost.solo,
        ),
    ]


def check_h10b(found: list[System]) -> Check:
    """The swarm is faster than every static split at least as accurate as it is."""
    swarm, *others = found
    rivals = [s for s in others if s.name.startswith("split") and s.accuracy >= swarm.accuracy]
    passed = all(swarm.throughput > s.throughput for s in rivals)
    named = ", ".join(s.name.split(":")[0] for s in rivals) or "none"
    detail = (
        f"swarm {swarm.accuracy:.3f} at {swarm.throughput:.1f}; static splits at least as "
        f"accurate: {named}"
    )
    return Check("H10b", passed, detail)


def confirmation(results: Path, config: SpeedConfig) -> str:
    """Markdown report of H8b and H10b for one confirmation run."""
    items = load_items(results, "gsm8k")
    found = systems(items, costs(config))
    lines = [f"# {results.name}", "", "| Condition | Accuracy |", "|---|---:|"]
    lines += [f"| {c} | {accuracy(rows):.4f} |" for c, rows in items.items()]
    lines += ["", "## H8b", ""]
    lines += [
        f"- **{c.hypothesis}: {'pass' if c.passed else 'fail'}.** {c.detail}"
        for c in check_h8b(items)
    ]
    lines += [
        "",
        "## H10b",
        "",
        "| System | Accuracy | Problems/s per answer token |",
        "|---|---:|---:|",
    ]
    lines += [f"| {s.name} | {s.accuracy:.3f} | {s.throughput:.1f} |" for s in found]
    check = check_h10b(found)
    lines += ["", f"**{check.hypothesis}: {'pass' if check.passed else 'fail'}.** {check.detail}"]
    return "\n".join(lines) + "\n"


def write_confirmation(results: Path, config: SpeedConfig, out: Path) -> None:
    """Write :func:`confirmation` to ``out/summary.md``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(confirmation(results, config), encoding="utf-8")
