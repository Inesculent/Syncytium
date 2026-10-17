"""The conditions a sweep evaluates.

Different conditions often describe the same weights: child k of N = 4 equals child 2k of
N = 8, and round-to-nearest equals the child with offset ½. :attr:`Condition.key` captures
this (bit width, group size, rotation, scaling and the exact set of grid offsets), so each
distinct model is evaluated once.
"""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations, product

from torch import Tensor

from syncytium.models import NamedTransform
from syncytium.quant import ChannelScales, Scheme, simulate_child, simulate_rebuild
from syncytium.sweep.config import GridConfig


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
        if s.rotation_seed is not None:
            setting += f"-r{s.rotation_seed}"
        if s.scaling:
            setting += "-s"
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
        s = self.scheme
        return (s.bits, s.group_size, s.rotation_seed, s.scaling, self.offsets)

    def transform(self, scales: ChannelScales | None = None) -> NamedTransform | None:
        """Weight transform for this condition; ``None`` means the original weights.

        Args:
            scales: Channel scales from calibration; required when the scheme scales.

        Raises:
            ValueError: If the scheme scales but no ``scales`` are given.
        """
        scheme, indices = self.scheme, self.indices
        if scheme is None:
            return None
        if scheme.scaling and scales is None:
            raise ValueError(f"{self.id} scales its channels but no calibration was given")

        def scales_of(name: str, weight: Tensor) -> Tensor | None:
            return scales(name, weight, scheme) if scheme.scaling else None

        if self.kind in (Kind.CHILD, Kind.RTN):
            return lambda name, w: simulate_child(w, scheme, indices[0], scales_of(name, w))
        return lambda name, w: simulate_rebuild(w, scheme, indices, scales=scales_of(name, w))


def plan(grid: GridConfig) -> list[Condition]:
    """Every condition in ``grid``, starting with the bf16 baseline."""
    conditions = [Condition(Kind.BF16)]
    settings = product(grid.bits, grid.group_sizes, grid.phases, grid.rotation, grid.scaling)
    for bits, group_size, phase, rotated, scaled in settings:
        seed = grid.seed if rotated else None
        for n in grid.children:
            scheme = Scheme(bits, group_size, n, phase, seed, scaled)
            conditions += _setting(scheme, grid.max_missing)
    baselines = product(grid.rtn_bits, grid.group_sizes, grid.rotation, grid.scaling)
    for bits, group_size, rotated, scaled in baselines:
        scheme = Scheme(bits, group_size, 1, 0.5, grid.seed if rotated else None, scaled)
        conditions.append(Condition(Kind.RTN, scheme, (0,)))
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
