"""Survival under device loss (``docs/preregistration-survival.md``, H12b).

Four devices; during a window each is lost independently with probability *p* and nothing is
restored. The swarm answers with every live member fused; two 6-bit pipelines, each stage held
by two devices and cross-routed, answer when every stage has a live copy. Every quantity is an
exact sum over the 16 device states, weighted by *p*^lost (1 - *p*)^live. Thresholds and rates
are the pre-registered ones and are not parameters.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from syncytium.eval import child_footprint
from syncytium.models import parameter_counts
from syncytium.speed.config import SpeedConfig
from syncytium.speed.elasticity import LARGE_BITS, costs
from syncytium.speed.network import Network, Swarm
from syncytium.speed.report import PASS_POINT, load_acceptance, outcome, worst_drafter
from syncytium.speed.rounds import model_rounds
from syncytium.tasks.analysis import Check, Items, accuracy, load_items

DEVICES = 4
RATES = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
PASS_RATES = (0.1, 0.2, 0.3)
LARGE = f"rtn/b{LARGE_BITS}"
# Stage 1 of both pipelines lives on devices 0 and 2, stage 2 on devices 1 and 3.
STAGES = ((0, 2), (1, 3))

State = tuple[bool, ...]
Answer = Callable[[State], float | None]


@dataclass(frozen=True)
class Point:
    """One system at one failure rate.

    Attributes:
        rate: Probability *p* that each device is lost.
        answered: Probability that a request can be answered.
        expected: Expected accuracy per request, unanswered requests counted as wrong.
    """

    rate: float
    answered: float
    expected: float

    @property
    def accuracy_if_answered(self) -> float:
        """Accuracy of the requests that are answered."""
        return self.expected / self.answered if self.answered else 0.0


def states() -> list[State]:
    """Every live/lost pattern of the four devices."""
    return list(product((True, False), repeat=DEVICES))


def weight(state: State, rate: float) -> float:
    """Probability of ``state`` when each device is lost with probability ``rate``."""
    live = sum(state)
    return (1 - rate) ** live * rate ** (DEVICES - live)


def swarm_answer(items: Items) -> Answer:
    """Accuracy of the live members fused together, or ``None`` if none is live."""

    def answer(state: State) -> float | None:
        live = [k for k in range(DEVICES) if state[k]]
        if not live:
            return None
        name = f"child/k{live[0]}" if len(live) == 1 else f"group/k{','.join(map(str, live))}"
        return accuracy(items[name])

    return answer


def pipelines_answer(items: Items) -> Answer:
    """The 6-bit model's accuracy if every stage has a live copy, else ``None``."""
    large = accuracy(items[LARGE])
    return lambda state: large if pipelines_up(state) else None


def pipelines_up(state: State) -> bool:
    """Whether every stage of the cross-routed pipelines has a live copy."""
    return all(any(state[d] for d in stage) for stage in STAGES)


def point(answer: Answer, rate: float) -> Point:
    """Probability of answering and expected accuracy at ``rate``."""
    answered = expected = 0.0
    for state in states():
        result = answer(state)
        if result is not None:
            answered += weight(state, rate)
            expected += weight(state, rate) * result
    return Point(rate, answered, expected)


def probability(condition: Callable[[State], bool], rate: float) -> float:
    """Probability that ``condition`` holds of the device state at ``rate``."""
    return sum(weight(s, rate) for s in states() if condition(s))


def neighbour_restorable(state: State) -> bool:
    """Whether neighbour bits can regrow every lost member: no two cyclically adjacent lost."""
    lost = [k for k in range(DEVICES) if not state[k]]
    if len(lost) == DEVICES:
        return False
    return not any((k + 1) % DEVICES in lost for k in lost)


def residue_restorable(tolerated: int) -> Callable[[State], bool]:
    """Whether a coded residue surviving ``tolerated`` losses can regrow every lost member."""
    return lambda state: sum(state) > 0 and DEVICES - sum(state) <= tolerated


def check_h12b(swarm: Sequence[Point], pipelines: Sequence[Point]) -> Check:
    """The swarm's expected per-request accuracy beats the pipelines' at every pass rate."""
    by_rate = {p.rate: p for p in pipelines}
    rows = [(s, by_rate[s.rate]) for s in swarm if s.rate in PASS_RATES]
    passed = all(s.expected > q.expected for s, q in rows)
    detail = "; ".join(f"p = {s.rate:g}: {s.expected:.4f} vs {q.expected:.4f}" for s, q in rows)
    return Check("H12b", passed, f"swarm vs replicated pipelines, {detail}")


@dataclass(frozen=True)
class Speeds:
    """Answer tokens per second on the pass-point network, for the throughput report.

    Attributes:
        fused: Tokens per second of a fused group of each size (1 is a lone child).
        pipeline: Tokens per second of one 6-bit pipeline over two devices, kept full.
    """

    fused: Mapping[int, float]
    pipeline: float


def speeds(config: SpeedConfig, network: Network = PASS_POINT) -> Speeds:
    """Speeds at the model size of ``config``, from H9's model and measurements."""
    scheme = config.setting.scheme
    child = child_footprint(parameter_counts(config.model.path), scheme).total_bits / 8
    rounds = model_rounds(config.model.path)
    cycles = {
        int(k): v
        for k, v in worst_drafter(load_acceptance(config.output))["tokens_per_cycle"].items()
    }
    fused = {1: network.memory_bandwidth / child}
    for size in range(2, DEVICES + 1):
        fused[size] = outcome(Swarm(rounds, child, size), network, cycles).fused
    cost = costs(config, network)
    return Speeds(fused, 2 / cost.pipeline(2))


def throughput(items: Items, speed: Speeds, rate: float) -> tuple[float, float]:
    """Expected correct answers per second per answer token: (swarm, pipelines)."""
    swarm_rate = pipe_rate = 0.0
    large = accuracy(items[LARGE])
    swarm = swarm_answer(items)
    for state in states():
        live = sum(state)
        if live:
            swarm_rate += weight(state, rate) * swarm(state) * speed.fused[live]
        complete = min(sum(state[d] for d in stage) for stage in STAGES)
        pipe_rate += weight(state, rate) * large * complete * speed.pipeline
    return swarm_rate, pipe_rate


def survival(results: Path, config: SpeedConfig) -> str:
    """Markdown report of H12b and the quantities reported alongside it."""
    items = load_items(results, "gsm8k")
    swarm = [point(swarm_answer(items), r) for r in RATES]
    pipes = [point(pipelines_answer(items), r) for r in RATES]
    speed = speeds(config)
    lines = [
        f"# {results.name}: survival under device loss (H12b)",
        "",
        f"6-bit model (`{LARGE}`): {accuracy(items[LARGE]):.4f} on GSM8K; full group "
        f"{accuracy(items['group/k0,1,2,3']):.4f}; bf16 {accuracy(items['bf16']):.4f}.",
        "",
        "| p | Swarm: answered | if answered | expected | Pipelines: answered | if answered "
        "| expected |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s, q in zip(swarm, pipes, strict=True):
        lines.append(
            f"| {s.rate:g} | {s.answered:.4f} | {s.accuracy_if_answered:.4f} "
            f"| **{s.expected:.4f}** | {q.answered:.4f} | {q.accuracy_if_answered:.4f} "
            f"| **{q.expected:.4f}** |"
        )
    check = check_h12b(swarm, pipes)
    lines += ["", f"**H12b: {'pass' if check.passed else 'fail'}.** {check.detail}", ""]
    lines += _secondary(items, speed)
    return "\n".join(lines) + "\n"


def write_survival(results: Path, config: SpeedConfig, out: Path) -> None:
    """Write :func:`survival` to ``out/survival.md``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "survival.md").write_text(survival(results, config), encoding="utf-8")


def _secondary(items: Items, speed: Speeds) -> list[str]:
    swarm = swarm_answer(items)
    lines = [
        "## Reported alongside",
        "",
        "| p | Correct answers/s per token: swarm | pipelines | Full health restorable: "
        "neighbour bits | residue t = 1 | t = 2 | t = 3 | pipelines | Swarm survivors' accuracy "
        "when neighbour bits cannot restore |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rate in RATES:
        fast, pipe = throughput(items, speed, rate)
        stuck = [s for s in states() if any(s) and not neighbour_restorable(s)]
        stuck_weight = sum(weight(s, rate) for s in stuck)
        stuck_accuracy = (
            sum(weight(s, rate) * swarm(s) for s in stuck) / stuck_weight if stuck_weight else 0.0
        )
        restore = [probability(neighbour_restorable, rate)]
        restore += [probability(residue_restorable(t), rate) for t in (1, 2, 3)]
        restore.append(probability(pipelines_up, rate))
        cells = " | ".join(f"{v:.4f}" for v in restore)
        lines.append(f"| {rate:g} | {fast:.2f} | {pipe:.2f} | {cells} | {stuck_accuracy:.4f} |")
    return lines
