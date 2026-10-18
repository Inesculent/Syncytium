"""The conditions a swarm task run evaluates (``docs/preregistration-swarm.md``).

Every child alone (a lone agent), every fused group of two or more children (H7, H11), the
bf16 model as the ceiling, for H13 the full group losing each member halfway through its
answers, and round-to-nearest single models at other bit widths (the static splits' large
model in ``docs/preregistration-disagreement.md``).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations

from torch import Tensor

from syncytium.models import NamedTransform
from syncytium.quant import ChannelScales, Estimator, Scheme, simulate_child, simulate_rebuild


class Kind(StrEnum):
    """What a condition evaluates."""

    BF16 = "bf16"
    CHILD = "child"  # one lone agent
    GROUP = "group"  # fused children; all of them is the rebuilt parent
    DROP = "drop"  # the full group, losing one member halfway through each answer
    RTN = "rtn"  # one round-to-nearest model at another bit width


@dataclass(frozen=True)
class TaskCondition:
    """One model state to evaluate.

    Attributes:
        kind: What is evaluated.
        indices: Children taking part (for a drop, the full group before the loss).
        dropped: The child lost halfway through each answer; only for :attr:`Kind.DROP`.
        bits: Bit width of a round-to-nearest model; only for :attr:`Kind.RTN`.
    """

    kind: Kind
    indices: tuple[int, ...] = ()
    dropped: int | None = None
    bits: int | None = None

    @property
    def id(self) -> str:
        """Stable identifier, unique within a run."""
        if self.kind == Kind.BF16:
            return self.kind.value
        if self.kind == Kind.DROP:
            return f"{self.kind.value}/k{self.dropped}"
        if self.kind == Kind.RTN:
            return f"{self.kind.value}/b{self.bits}"
        return f"{self.kind.value}/k{','.join(map(str, self.indices))}"

    @property
    def survivors(self) -> "TaskCondition":
        """The group left after a drop.

        Raises:
            ValueError: If this condition is not a drop.
        """
        if self.kind != Kind.DROP:
            raise ValueError(f"{self.id} drops no child")
        kept = tuple(i for i in self.indices if i != self.dropped)
        return TaskCondition(Kind.GROUP, kept)

    def transform(
        self, scheme: Scheme, estimator: Estimator, scales: ChannelScales | None = None
    ) -> NamedTransform | None:
        """Weights of a child or group; ``None`` for bf16 (the original weights).

        A group of every child is the exact rebuild; a smaller group uses ``estimator``. A
        round-to-nearest model shares the children's group size, rotation and scaling.

        Raises:
            ValueError: For a drop, which has two weight states (see :attr:`survivors`), or if
                the scheme scales but no ``scales`` are given.
        """
        if self.kind == Kind.BF16:
            return None
        if self.kind == Kind.DROP:
            raise ValueError(f"{self.id} has two weight states; transform its groups instead")
        if scheme.scaling and scales is None:
            raise ValueError(f"{self.id} scales its channels but no calibration was given")
        indices = self.indices
        if self.kind == Kind.RTN:
            # The sweeps' round-to-nearest baseline: a single "child" with offset ½.
            scheme = replace(scheme, bits=self.bits, num_children=1, phase=0.5)
            indices = (0,)

        def scales_of(name: str, weight: Tensor) -> Tensor | None:
            return scales(name, weight, scheme) if scheme.scaling else None

        if self.kind in (Kind.CHILD, Kind.RTN):
            return lambda name, w: simulate_child(w, scheme, indices[0], scales_of(name, w))
        return lambda name, w: simulate_rebuild(w, scheme, indices, estimator, scales_of(name, w))


def plan(
    num_children: int, drop: bool, full_only: bool = False, rtn_bits: Sequence[int] = ()
) -> list[TaskCondition]:
    """Every condition for ``num_children`` children, in the order a run evaluates them.

    With ``full_only`` the full group is the only group; ``rtn_bits`` adds round-to-nearest
    models. The full group comes before the drops, which reuse its answers.
    """
    everyone = tuple(range(num_children))
    conditions = [TaskCondition(Kind.BF16)]
    conditions += [TaskCondition(Kind.CHILD, (k,)) for k in everyone]
    smallest = num_children if full_only else 2
    for size in range(num_children, smallest - 1, -1):
        conditions += [TaskCondition(Kind.GROUP, kept) for kept in combinations(everyone, size)]
    conditions += [TaskCondition(Kind.RTN, bits=b) for b in rtn_bits]
    if drop:
        conditions += [TaskCondition(Kind.DROP, everyone, k) for k in everyone]
    return conditions
