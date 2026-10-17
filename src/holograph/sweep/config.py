"""Sweep configuration, read from TOML files in ``configs/``."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Which model to evaluate and where to run it."""

    path: str
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass(frozen=True)
class EvalConfig:
    """How perplexity is measured."""

    window: int = 2048
    max_windows: int | None = None


@dataclass(frozen=True)
class GridConfig:
    """The conditions to evaluate.

    Every combination of ``bits``, ``group_sizes``, ``phases`` and ``rotation`` is one
    setting. Each setting evaluates, for every N in ``children``: each child alone, the full
    rebuild, and every partial rebuild missing up to ``max_missing`` children. ``rtn_bits``
    adds standard round-to-nearest baselines at each listed bit width. Rotated settings use
    the shared rotation drawn from ``seed``.
    """

    bits: tuple[int, ...]
    group_sizes: tuple[int, ...]
    children: tuple[int, ...]
    phases: tuple[float, ...] = (0.0,)
    rotation: tuple[bool, ...] = (False,)
    seed: int = 0
    max_missing: int = 1
    rtn_bits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("bits", "group_sizes", "children", "phases", "rotation"):
            if not getattr(self, name):
                raise ValueError(f"grid.{name} must not be empty")
        if self.max_missing < 0:
            raise ValueError(f"grid.max_missing must be non-negative, got {self.max_missing}")


@dataclass(frozen=True)
class SweepConfig:
    """A complete sweep: model, evaluation, grid and output directory."""

    name: str
    output: Path
    model: ModelConfig
    evaluation: EvalConfig
    grid: GridConfig


def load_config(path: Path) -> SweepConfig:
    """Read and validate a sweep config. Unknown keys are rejected."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    return SweepConfig(
        name=raw["name"],
        output=Path(raw["output"]),
        model=ModelConfig(**raw["model"]),
        evaluation=EvalConfig(**raw.get("eval", {})),
        grid=GridConfig(**_as_tuples(raw["grid"])),
    )


def _as_tuples(table: dict[str, Any]) -> dict[str, Any]:
    """Convert TOML arrays to tuples so configs stay immutable."""
    return {key: tuple(value) if isinstance(value, list) else value for key, value in table.items()}
