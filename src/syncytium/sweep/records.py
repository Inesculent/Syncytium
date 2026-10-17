"""Sweep outputs: an append-only results log and per-invocation run metadata."""

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

import syncytium


class ResultLog:
    """Append-only JSONL file with one row per evaluated condition.

    Rows are written as soon as each condition finishes, so an interrupted sweep resumes
    where it stopped.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def completed(self) -> dict[str, dict[str, Any]]:
        """Rows already written, keyed by condition id."""
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        return {row["id"]: row for row in rows}

    def append(self, row: dict[str, Any]) -> None:
        """Write one row and flush it to disk."""
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")


def append_run_metadata(path: Path, config_path: Path) -> None:
    """Record what produced this run: config, code version, packages and hardware."""
    record = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config_path.read_text(encoding="utf-8"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "syncytium": syncytium.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def _git(*args: str) -> str | None:
    """Output of a git command, or ``None`` outside a git checkout."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
