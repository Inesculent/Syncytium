"""The conditions a swarm task run evaluates (``docs/preregistration-swarm.md``).

Every child alone (a lone agent), every fused group of two or more children (H7, H11), the
bf16 model as the ceiling, and for H13 the full group losing each member halfway through its
answers.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class TaskCondition:
    """One model state to evaluate.

    Attributes:
        kind: What is evaluated.
        indices: Children taking part (for a drop, the full group before the loss).
        dropped: The child lost halfway through each answer; only for :attr:`Kind.DROP`.
    """

    kind: Kind
    indices: tuple[int, ...] = ()
    dropped: int | None = None

    @property
    def id(self) -> str:
        """Stable identifier, unique within a run."""
        if self.kind == Kind.BF16:
            return self.kind.value
        if self.kind == Kind.DROP:
            return f"{self.kind.value}/k{self.dropped}"
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

        A group of every child is the exact rebuild; a smaller group uses ``estimator``.

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

        def scales_of(name: str, weight: Tensor) -> Tensor | None:
            return scales(name, weight, scheme) if scheme.scaling else None

        if self.kind == Kind.CHILD:
            return lambda name, w: simulate_child(w, scheme, indices[0], scales_of(name, w))
        return lambda name, w: simulate_rebuild(w, scheme, indices, estimator, scales_of(name, w))


def plan(num_children: int, drop: bool) -> list[TaskCondition]:
    """Every condition for ``num_children`` children, in the order a run evaluates them.

    The full group comes before the drops, which reuse its answers.
    """
    everyone = tuple(range(num_children))
    conditions = [TaskCondition(Kind.BF16)]
    conditions += [TaskCondition(Kind.CHILD, (k,)) for k in everyone]
    for size in range(num_children, 1, -1):
        conditions += [TaskCondition(Kind.GROUP, kept) for kept in combinations(everyone, size)]
    if drop:
        conditions += [TaskCondition(Kind.DROP, everyone, k) for k in everyone]
    return conditions
