"""Speed-model run configuration (H9), read from TOML files in ``configs/``."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from syncytium.sweep.config import EvalConfig, ModelConfig
from syncytium.tasks.config import SwarmSetting


@dataclass(frozen=True)
class AcceptanceConfig:
    """Where the drafters' agreement with the fused group is measured.

    ``windows`` non-overlapping windows of ``window`` tokens from the WikiText-2 test split,
    and every draft length in ``drafts``.
    """

    windows: int = 16
    window: int = 2048
    drafts: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

    def __post_init__(self) -> None:
        if not self.drafts or min(self.drafts) < 1:
            raise ValueError(f"acceptance.drafts must be positive, got {self.drafts}")


@dataclass(frozen=True)
class SpeedConfig:
    """A complete speed-model run: model, children, acceptance text and output directory."""

    name: str
    output: Path
    model: ModelConfig
    evaluation: EvalConfig
    setting: SwarmSetting
    acceptance: AcceptanceConfig


def load_speed_config(path: Path) -> SpeedConfig:
    """Read and validate a speed config. Unknown keys are rejected."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    acceptance = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw.get("acceptance", {}).items()
    }
    return SpeedConfig(
        name=raw["name"],
        output=Path(raw["output"]),
        model=ModelConfig(**raw["model"]),
        evaluation=EvalConfig(**raw.get("eval", {})),
        setting=SwarmSetting(**raw["setting"]),
        acceptance=AcceptanceConfig(**acceptance),
    )
