"""Pooled-pipeline sweep configuration, read from TOML files in ``configs/``."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from syncytium.quant import Estimator, Scheme
from syncytium.sweep.config import EvalConfig, ModelConfig


@dataclass(frozen=True)
class PoolGridConfig:
    """The device grid and the losses to evaluate.

    The parent is split into stage 0 (embedding and output head) plus ``stages`` blocks of
    decoder layers; each stage is held by ``children`` devices with one child each. The
    sweep evaluates every single-device loss, every stage reduced to one child, and
    ``samples`` random patterns for each number of lost devices from 2 to ``max_lost``,
    under each estimator in ``estimators``. Rotated grids use the rotation drawn from
    ``seed``; ``pattern_seed`` draws the random patterns.
    """

    bits: int
    group_size: int
    children: int
    stages: int
    phase: float = 0.0
    rotation: bool = False
    seed: int = 0
    estimators: tuple[str, ...] = ("mean", "midpoint")
    max_lost: int = 4
    samples: int = 8
    pattern_seed: int = 0

    def __post_init__(self) -> None:
        if self.children < 2:
            raise ValueError(f"pool.children must be at least 2, got {self.children}")
        if self.stages < 1:
            raise ValueError(f"pool.stages must be positive, got {self.stages}")
        if not self.estimators:
            raise ValueError("pool.estimators must not be empty")
        for name in self.estimators:
            Estimator(name)  # rejects unknown names
        # Beyond this many losses some stage must lose every child.
        most = (self.stages + 1) * (self.children - 1)
        if not 1 <= self.max_lost <= most:
            raise ValueError(f"pool.max_lost must be in [1, {most}], got {self.max_lost}")
        if self.samples < 0:
            raise ValueError(f"pool.samples must be non-negative, got {self.samples}")

    @property
    def scheme(self) -> Scheme:
        """The quantization scheme shared by every stage."""
        seed = self.seed if self.rotation else None
        return Scheme(self.bits, self.group_size, self.children, self.phase, seed)


@dataclass(frozen=True)
class PoolConfig:
    """A complete pooled-pipeline sweep: model, evaluation, grid and output directory."""

    name: str
    output: Path
    model: ModelConfig
    evaluation: EvalConfig
    pool: PoolGridConfig


def load_pool_config(path: Path) -> PoolConfig:
    """Read and validate a pooled-pipeline config. Unknown keys are rejected."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    pool = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw["pool"].items()
    }
    return PoolConfig(
        name=raw["name"],
        output=Path(raw["output"]),
        model=ModelConfig(**raw["model"]),
        evaluation=EvalConfig(**raw.get("eval", {})),
        pool=PoolGridConfig(**pool),
    )
