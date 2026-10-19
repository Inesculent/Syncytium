"""Tokens per second of solo and fused agents under a stated network model (H9).

Assumptions, recorded in ``docs/preregistration-swarm.md`` before any acceptance rate was
measured:

- Decoding is memory-bound: a forward pass over a few tokens costs one read of the child's
  weights at the device's memory bandwidth. Compute and KV-cache reads are ignored.
- Every device has a full-duplex link of the stated bandwidth to a switch, and each step of
  a collective costs one latency. Communication does not overlap computation.
- Activations travel in bf16. Each round is an all-reduce, done by whichever is faster of a
  direct exchange (one step; each device sends its vector to every other) and a
  reduce-scatter followed by an all-gather (two steps; each sends 2(N - 1)/N of it).
- The output logits only need their argmax: reduce-scatter them, then exchange each slice's
  best value and index.
- Speculative decoding: the requesting agent drafts k tokens alone (k solo passes); the fused
  group then checks them in one pass over k + 1 positions, which yields the accepted drafts
  and one more token of its own.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from syncytium.speed.rounds import Round

BYTES_PER_VALUE = 2  # bf16 activations
BEST_BYTES = 8  # a slice's best logit (fp32) and its index (int32)


@dataclass(frozen=True)
class Network:
    """The hardware every device runs on.

    Attributes:
        latency: Seconds per step of a collective.
        bandwidth: Link bandwidth in bits per second, each way.
        memory_bandwidth: Device memory bandwidth in bytes per second.
    """

    latency: float
    bandwidth: float
    memory_bandwidth: float


@dataclass(frozen=True)
class Swarm:
    """What one fused group of devices runs.

    Attributes:
        rounds: The rounds of one token's forward pass, from
            :func:`~syncytium.speed.rounds.trace_rounds`.
        child_bytes: Weight memory of one child.
        devices: Devices in the fused group, N.
    """

    rounds: Sequence[Round]
    child_bytes: float
    devices: int


def solo_seconds(swarm: Swarm, network: Network) -> float:
    """Seconds per token for one device decoding alone."""
    return swarm.child_bytes / network.memory_bandwidth


def all_reduce_seconds(values: int, devices: int, network: Network) -> float:
    """Seconds to average ``values`` across ``devices``: the faster of the two algorithms."""
    bits = values * BYTES_PER_VALUE * 8
    direct = network.latency + (devices - 1) * bits / network.bandwidth
    two_step = 2 * network.latency + 2 * (devices - 1) / devices * bits / network.bandwidth
    return min(direct, two_step)


def argmax_seconds(values: int, devices: int, network: Network) -> float:
    """Seconds to find the argmax of logits averaged across ``devices``."""
    bits = values * BYTES_PER_VALUE * 8
    scatter = network.latency + (devices - 1) / devices * bits / network.bandwidth
    gather = network.latency + (devices - 1) * BEST_BYTES * 8 / network.bandwidth
    return min(all_reduce_seconds(values, devices, network), scatter + gather)


def leader_reduce_seconds(values: int, devices: int, network: Network) -> float:
    """Seconds for the leader reduce of the live system (H18).

    Every member sends its vector to the leader, which returns the mean to each. The leader's
    link receives (n - 1) vectors and sends (n - 1), one after another; each direction adds
    one latency in flight.
    """
    bits = values * BYTES_PER_VALUE * 8
    return 2 * network.latency + 2 * (devices - 1) * bits / network.bandwidth


def fused_pass_seconds(swarm: Swarm, positions: int, network: Network) -> float:
    """Seconds for the fused group to run one forward pass over ``positions`` tokens."""
    talk = sum(
        (argmax_seconds if r.head else all_reduce_seconds)(
            r.values * positions, swarm.devices, network
        )
        for r in swarm.rounds
    )
    return solo_seconds(swarm, network) + talk


def fused_tokens_per_second(
    swarm: Swarm, network: Network, draft: int = 0, tokens_per_cycle: float = 1.0
) -> float:
    """Tokens per second of the fused group, with ``draft`` speculative tokens per cycle.

    Args:
        swarm: The fused group.
        network: The hardware.
        draft: Tokens the requesting agent drafts per cycle; 0 for no speculation.
        tokens_per_cycle: Tokens produced per cycle (accepted drafts plus the group's own),
            from :func:`~syncytium.speed.acceptance.tokens_per_cycle`; 1 without drafts.
    """
    if draft == 0:
        return 1 / fused_pass_seconds(swarm, 1, network)
    cycle = draft * solo_seconds(swarm, network) + fused_pass_seconds(swarm, draft + 1, network)
    return tokens_per_cycle / cycle
