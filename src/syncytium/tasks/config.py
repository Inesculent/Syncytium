"""Swarm task-run configuration, read from TOML files in ``configs/``."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from syncytium.quant import Estimator, Scheme
from syncytium.sweep.config import EvalConfig, ModelConfig


@dataclass(frozen=True)
class SwarmSetting:
    """The children every condition is built from.

    ``estimator`` is how groups smaller than the full set combine their children; the
    pre-registration fixes it as "mean" unless ``pool-2b`` shows the midpoint better. With
    ``drop``, the full group also loses each member halfway through its answers (H13).
    """

    bits: int
    group_size: int
    children: int
    phase: float = 0.0
    rotation: bool = False
    scaling: bool = False
    seed: int = 0
    estimator: str = "mean"
    drop: bool = True

    def __post_init__(self) -> None:
        if self.children < 2:
            raise ValueError(f"setting.children must be at least 2, got {self.children}")
        Estimator(self.estimator)  # rejects unknown names

    @property
    def scheme(self) -> Scheme:
        """The quantization scheme shared by every child."""
        seed = self.seed if self.rotation else None
        return Scheme(self.bits, self.group_size, self.children, self.phase, seed, self.scaling)


@dataclass(frozen=True)
class GSM8KConfig:
    """How GSM8K is run. The test split is first run by the pre-registered configuration."""

    split: str = "test"
    limit: int | None = None
    batch_size: int = 32
    max_new_tokens: int = 256


@dataclass(frozen=True)
class MMLUConfig:
    """How MMLU is run. Worked examples always come from the dev split."""

    split: str = "test"
    limit: int | None = None
    batch_size: int = 16


@dataclass(frozen=True)
class TaskConfig:
    """A complete task run: model, calibration, children, tasks and output directory."""

    name: str
    output: Path
    model: ModelConfig
    evaluation: EvalConfig
    setting: SwarmSetting
    gsm8k: GSM8KConfig | None
    mmlu: MMLUConfig | None


def load_task_config(path: Path) -> TaskConfig:
    """Read and validate a task config. Unknown keys are rejected.

    Raises:
        ValueError: If the config names neither task.
    """
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if "gsm8k" not in raw and "mmlu" not in raw:
        raise ValueError(f"{path} names no task; add a [gsm8k] or [mmlu] table")
    return TaskConfig(
        name=raw["name"],
        output=Path(raw["output"]),
        model=ModelConfig(**raw["model"]),
        evaluation=EvalConfig(**raw.get("eval", {})),
        setting=SwarmSetting(**raw["setting"]),
        gsm8k=GSM8KConfig(**raw["gsm8k"]) if "gsm8k" in raw else None,
        mmlu=MMLUConfig(**raw["mmlu"]) if "mmlu" in raw else None,
    )
