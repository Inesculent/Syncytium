"""The device states a pooled-pipeline sweep evaluates.

A state lists, for each stage, the children whose devices are alive. States that give every
stage the same effective weights share a :attr:`PoolCondition.key` and are evaluated once:
when a stage has all of its children or just one, both estimators give the same weights.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from torch import Tensor

from syncytium.models import NamedTransform
from syncytium.pool.config import PoolGridConfig
from syncytium.quant import ChannelScales, Estimator, Scheme, simulate_rebuild

State = tuple[frozenset[int], ...]

# Draws per wanted pattern before giving up on finding more distinct valid ones.
ATTEMPTS_PER_SAMPLE = 100


class Kind(StrEnum):
    """What a state represents."""

    BF16 = "bf16"
    FULL = "full"  # every device alive: the rebuilt parent
    CHILD = "child"  # every stage on the same single child: one child model
    DEVICE = "device"  # one device lost
    STAGE = "stage"  # one stage reduced to a single child
    RANDOM = "random"  # devices lost at random across the grid
    COMPONENT = "component"  # one stage of a random pattern on its own (additivity check)


@dataclass(frozen=True)
class PoolCondition:
    """One state of the device grid to evaluate.

    Attributes:
        kind: What the state represents.
        scheme: Quantization settings; ``None`` only for the bf16 baseline.
        alive: For each stage, the indices of the children whose devices are alive.
        estimator: How a stage with some children missing estimates its weights.
    """

    kind: Kind
    scheme: Scheme | None = None
    alive: State = ()
    estimator: Estimator = Estimator.MEAN

    @property
    def lost(self) -> list[tuple[int, int]]:
        """Lost devices as (stage, child) pairs."""
        if self.scheme is None:
            return []
        children = range(self.scheme.num_children)
        return [(s, k) for s, alive in enumerate(self.alive) for k in children if k not in alive]

    @property
    def estimator_matters(self) -> bool:
        """Whether some stage has more than one but not all of its children."""
        if self.scheme is None:
            return False
        return any(1 < len(alive) < self.scheme.num_children for alive in self.alive)

    @property
    def id(self) -> str:
        """Stable, human-readable identifier, unique within a sweep."""
        if self.scheme is None:
            return self.kind.value
        lost = "+".join(f"{stage}.{child}" for stage, child in self.lost) or "none"
        if self.estimator_matters:
            return f"{self.kind.value}/{self.estimator.value}/lost-{lost}"
        return f"{self.kind.value}/lost-{lost}"

    @property
    def key(self) -> tuple:
        """States with equal keys produce identical weights."""
        if self.scheme is None:
            return (self.kind.value,)
        n = self.scheme.num_children
        return tuple(
            (alive, self.estimator if 1 < len(alive) < n else None) for alive in self.alive
        )

    def transform(
        self, stages: Mapping[str, int], scales: ChannelScales | None = None
    ) -> NamedTransform | None:
        """Weight transform for this state; ``None`` means the original weights.

        Args:
            stages: Stage of each weight, from :func:`~syncytium.pool.layout.assign_stages`.
            scales: Channel scales from calibration; required when the scheme scales.

        Raises:
            ValueError: If the scheme scales but no ``scales`` are given.
        """
        scheme, alive, estimator = self.scheme, self.alive, self.estimator
        if scheme is None:
            return None
        if scheme.scaling and scales is None:
            raise ValueError(f"{self.id} scales its channels but no calibration was given")

        def transform(name: str, weight: Tensor) -> Tensor:
            channel = scales(name, weight, scheme) if scheme.scaling else None
            return simulate_rebuild(weight, scheme, alive[stages[name]], estimator, channel)

        return transform


def plan(grid: PoolGridConfig) -> list[PoolCondition]:
    """Every state in ``grid``, starting with the bf16 baseline and the full pool."""
    scheme, n, count = grid.scheme, grid.children, grid.stages + 1
    everyone = frozenset(range(n))
    estimators = [Estimator(name) for name in grid.estimators]

    def state(changes: Mapping[int, frozenset[int]]) -> State:
        return tuple(changes.get(stage, everyone) for stage in range(count))

    conditions = [PoolCondition(Kind.BF16), PoolCondition(Kind.FULL, scheme, state({}))]
    conditions += [PoolCondition(Kind.CHILD, scheme, (frozenset({k}),) * count) for k in range(n)]
    devices = list(product(range(count), range(n)))
    for estimator in estimators:
        conditions += [
            PoolCondition(Kind.DEVICE, scheme, state({s: everyone - {k}}), estimator)
            for s, k in devices
        ]
    conditions += [
        PoolCondition(Kind.STAGE, scheme, state({s: frozenset({k})})) for s, k in devices
    ]
    patterns = random_patterns(grid)
    for estimator in estimators:
        for pattern in patterns:
            conditions.append(PoolCondition(Kind.RANDOM, scheme, pattern, estimator))
            conditions += [
                PoolCondition(Kind.COMPONENT, scheme, state({s: alive}), estimator)
                for s, alive in enumerate(pattern)
                if alive != everyone
            ]
    return list({condition.id: condition for condition in conditions}.values())


def random_patterns(grid: PoolGridConfig) -> list[State]:
    """Up to ``samples`` distinct loss patterns per number of lost devices, 2 … ``max_lost``.

    Devices fail uniformly at random. Patterns that leave a stage with no child are drawn
    again, since the pipeline cannot run without every stage.
    """
    rng = random.Random(grid.pattern_seed)
    n, count = grid.children, grid.stages + 1
    devices = list(product(range(count), range(n)))
    patterns: list[State] = []
    for lost_count in range(2, grid.max_lost + 1):
        drawn: list[State] = []
        for _ in range(ATTEMPTS_PER_SAMPLE * grid.samples):
            if len(drawn) == grid.samples:
                break
            lost = set(rng.sample(devices, lost_count))
            pattern = tuple(
                frozenset(k for k in range(n) if (s, k) not in lost) for s in range(count)
            )
            if all(pattern) and pattern not in drawn:
                drawn.append(pattern)
        patterns += drawn
    return patterns
