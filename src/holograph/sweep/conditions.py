"""The conditions a sweep evaluates.

Different conditions often describe the same weights: child k of N = 4 equals child 2k of
N = 8, and round-to-nearest equals the child with offset ½. :attr:`Condition.key` captures
this (bit width, group size and the exact set of grid offsets), so each distinct model is
evaluated once.
"""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations, product

from holograph.models import Transform
from holograph.quant import Scheme, simulate_child, simulate_rebuild
from holograph.sweep.config import GridConfig


class Kind(StrEnum):
    """What a condition evaluates."""

    BF16 = "bf16"
    CHILD = "child"
    REBUILD = "rebuild"
    PARTIAL = "partial"
    RTN = "rtn"


@dataclass(frozen=True)
class Condition:
    """One model variant to evaluate.

    Attributes:
        kind: What is evaluated.
        scheme: Quantization settings; ``None`` only for the bf16 baseline.
        indices: Child indices used (one for a child, several for a rebuild).
    """

    kind: Kind
    scheme: Scheme | None = None
    indices: tuple[int, ...] = ()

    @property
    def id(self) -> str:
        """Stable, human-readable identifier, unique within a sweep."""
        if self.scheme is None:
            return self.kind.value
        s = self.scheme
        setting = f"b{s.bits}-g{s.group_size}-n{s.num_children}-p{s.phase:g}"
        return f"{self.kind.value}/{setting}/k{','.join(map(str, self.indices))}"

    @property
    def offsets(self) -> frozenset[Fraction]:
        """Exact grid offsets (k + φ)/N of the children used."""
        if self.scheme is None:
            return frozenset()
        phase, n = Fraction(self.scheme.phase), self.scheme.num_children
        return frozenset((index + phase) / n for index in self.indices)

    @property
    def key(self) -> tuple:
        """Conditions with equal keys produce identical weights."""
        if self.scheme is None:
            return (self.kind.value,)
        return (self.scheme.bits, self.scheme.group_size, self.offsets)

    def transform(self) -> Transform | None:
        """Weight transform for this condition; ``None`` means the original weights."""
        scheme, indices = self.scheme, self.indices
        if scheme is None:
            return None
        if self.kind in (Kind.CHILD, Kind.RTN):
            return lambda weight: simulate_child(weight, scheme, indices[0])
        return lambda weight: simulate_rebuild(weight, scheme, indices)


def plan(grid: GridConfig) -> list[Condition]:
    """Every condition in ``grid``, starting with the bf16 baseline."""
    conditions = [Condition(Kind.BF16)]
    for bits, group_size, phase in product(grid.bits, grid.group_sizes, grid.phases):
        for n in grid.children:
            conditions += _setting(Scheme(bits, group_size, n, phase), grid.max_missing)
    for bits, group_size in product(grid.rtn_bits, grid.group_sizes):
        conditions.append(Condition(Kind.RTN, Scheme(bits, group_size, 1, phase=0.5), (0,)))
    return conditions


def _setting(scheme: Scheme, max_missing: int) -> list[Condition]:
    """Children, full rebuild and partial rebuilds for one scheme."""
    n = scheme.num_children
    found = [Condition(Kind.CHILD, scheme, (k,)) for k in range(n)]
    found.append(Condition(Kind.REBUILD, scheme, tuple(range(n))))
    for missing in range(1, min(max_missing, n - 1) + 1):
        found += [
            Condition(Kind.PARTIAL, scheme, kept) for kept in combinations(range(n), n - missing)
        ]
    return found
