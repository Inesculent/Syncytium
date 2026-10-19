"""Which runs exist, which are finished, and which have been copied back.

Every config in ``configs/`` is matched with its output directory: how many of the conditions
it plans are logged, which commit last ran it, and whether its report is in ``docs/results/``.
Results directories that no config writes to are listed too, so stray copies stand out.
"""

import json
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from syncytium.pool.conditions import plan as pool_plan
from syncytium.pool.config import load_pool_config
from syncytium.speed.config import load_speed_config
from syncytium.speed.runner import ACCEPTANCE
from syncytium.sweep.conditions import plan as sweep_plan
from syncytium.sweep.config import load_config
from syncytium.tasks.conditions import Kind as TaskKind
from syncytium.tasks.conditions import plan as task_plan
from syncytium.tasks.config import load_task_config


class Kind(StrEnum):
    """What a config runs, decided by its tables as in ``slurm/sweep.sbatch``."""

    SWEEP = "sweep"
    POOL = "pool"
    SPEED = "speed"
    TASKS = "tasks"


class State(StrEnum):
    """How far a run has got, as far as the local results show."""

    MISSING = "not here"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Run:
    """One config and what its local results contain.

    Attributes:
        name: The config's name, also its results directory.
        kind: What it runs.
        purpose: The first sentence of the config's opening comment.
        expected: Rows its plan will log when finished.
        logged: Distinct rows logged locally.
        commit: Short commit of the latest invocation, if any.
        started: Start time of the latest invocation, if any.
        job: SLURM job id of the latest invocation, which names its log, if any.
        reported: Whether ``docs/results/<name>/`` exists.
    """

    name: str
    kind: Kind
    purpose: str
    expected: int
    logged: int
    commit: str | None
    started: str | None
    job: str | None
    reported: bool

    @property
    def state(self) -> State:
        """Missing, partial or complete."""
        if self.logged == 0:
            return State.MISSING
        return State.COMPLETE if self.logged >= self.expected else State.PARTIAL


def kind_of(path: Path) -> Kind:
    """What a config runs, from the tables it has."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if "pool" in raw:
        return Kind.POOL
    if "acceptance" in raw:
        return Kind.SPEED
    if "setting" in raw:
        return Kind.TASKS
    return Kind.SWEEP


def expected_rows(path: Path, kind: Kind) -> tuple[Path, int]:
    """The output directory and number of rows a finished run of ``path`` logs."""
    if kind == Kind.POOL:
        pool = load_pool_config(path)
        return pool.output, len(pool_plan(pool.pool))
    if kind == Kind.SPEED:
        speed = load_speed_config(path)
        return speed.output, speed.setting.children
    if kind == Kind.TASKS:
        task = load_task_config(path)
        s = task.setting
        conditions = task_plan(s.children, s.drop, s.full_only, s.rtn_bits)
        rows = len(conditions) if task.gsm8k else 0
        if task.mmlu:
            rows += sum(c.kind != TaskKind.DROP for c in conditions)
        return task.output, rows
    sweep = load_config(path)
    return sweep.output, len(sweep_plan(sweep.grid))


def purpose(path: Path) -> str:
    """The first sentence of the comment a config opens with."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("# ").strip())
    text = " ".join(lines)
    return text.split(". ")[0].rstrip(".") if text else ""


def inspect(path: Path, root: Path) -> Run:
    """The status of the run that ``path`` configures, with paths relative to ``root``."""
    kind = kind_of(path)
    output, expected = expected_rows(path, kind)
    directory = root / output
    log = directory / (ACCEPTANCE if kind == Kind.SPEED else "results.jsonl")
    logged = len({row["id"] for row in _rows(log)})
    runs = _rows(directory / "runs.jsonl")
    last = runs[-1] if runs else {}
    commit = last.get("git_commit")
    return Run(
        name=path.stem,
        kind=kind,
        purpose=purpose(path),
        expected=expected,
        logged=logged,
        commit=commit[:7] if commit else None,
        started=last.get("started"),
        job=last.get("slurm_job"),
        reported=(root / "docs" / "results" / path.stem).exists(),
    )


def inventory(root: Path) -> tuple[list[Run], list[str]]:
    """Every config's run, and results directories that no config writes to."""
    runs = [inspect(path, root) for path in sorted((root / "configs").glob("*.toml"))]
    known = {run.name for run in runs}
    results = root / "results"
    strays = (
        sorted(d.name for d in results.iterdir() if d.is_dir() and d.name not in known)
        if results.exists()
        else []
    )
    return runs, strays


def table(runs: list[Run], strays: list[str]) -> str:
    """A plain-text table of ``runs``, then the stray results directories."""
    header = f"{'run':<18} {'kind':<6} {'state':<9} {'rows':>9}  {'commit':<7}  {'job':<9}"
    lines = [f"{header}  {'report':<6}  purpose"]
    for run in runs:
        rows = f"{run.logged}/{run.expected}"
        report = "yes" if run.reported else "-"
        lines.append(
            f"{run.name:<18} {run.kind:<6} {run.state:<9} {rows:>9}  {run.commit or '-':<7}  "
            f"{run.job or '-':<9}  {report:<6}  {_shorten(run.purpose, 60)}"
        )
    if strays:
        lines += ["", "Results with no config: " + ", ".join(strays)]
    return "\n".join(lines)


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
