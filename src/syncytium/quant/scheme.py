"""Settings shared by one set of staggered children."""

from dataclasses import dataclass

from syncytium.quant.grid import MAX_BITS

GRID_STORAGE = ("float16", "float32")


@dataclass(frozen=True)
class Scheme:
    """Quantization settings shared by every child in a set.

    Attributes:
        bits: Bits per child code, b.
        group_size: Weights per quantization group, g.
        num_children: Number of children, N.
        phase: Offset phase φ; child k uses offset (k + φ)/N.
        rotation_seed: Seed of the shared group rotation, or ``None`` for no rotation.
        scaling: Whether input channels are scaled by searched activation-aware scales
            before quantization (see :mod:`syncytium.quant.scaling`).
        grid_storage: Dtype shard files store each group's minimum and step in, "float16" or
            "float32". Simulation always fits the grid in float32. Channel scales can push
            scaled weights past fp16's range (they do at 4B), and such a set needs float32.
    """

    bits: int
    group_size: int
    num_children: int
    phase: float = 0.0
    rotation_seed: int | None = None
    scaling: bool = False
    grid_storage: str = "float16"

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= MAX_BITS:
            raise ValueError(f"bits must be in [1, {MAX_BITS}], got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.num_children < 1:
            raise ValueError(f"num_children must be positive, got {self.num_children}")
        if not 0 <= self.phase < 1:
            raise ValueError(f"phase must be in [0, 1), got {self.phase}")
        if self.grid_storage not in GRID_STORAGE:
            raise ValueError(f"grid_storage must be one of {GRID_STORAGE}, got {self.grid_storage}")
        if self.rotation_seed is not None and self.group_size & (self.group_size - 1):
            raise ValueError(f"rotation needs a power-of-two group_size, got {self.group_size}")
