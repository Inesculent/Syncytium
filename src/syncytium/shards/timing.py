"""Timing a rebuild from shard files: the "rebuild wall-clock time" measurement.

Each timed rebuild is appended to ``rebuilds.jsonl`` next to the shards, so repeated runs and
different child subsets accumulate in one place.
"""

import json
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from syncytium.shards.manifest import MANIFEST_NAME, Manifest, ShardError, shard_name
from syncytium.shards.reader import combine_shards, verify_shard

LOG_NAME = "rebuilds.jsonl"


@dataclass(frozen=True)
class RebuildTiming:
    """Wall-clock cost of rebuilding weights from shard files.

    Attributes:
        children: Indices of the children used.
        exact: Whether every child was present, so the rebuild was checked against the
            parent hash; otherwise it was a partial rebuild.
        dtype: Dtype of the rebuilt weights.
        bytes_read: Total size of the shard files read.
        verify_seconds: Checking every file against its hash in the manifest.
        rebuild_seconds: Streaming the codes from the files, combining them (and checking the
            parent hash) and dequantizing.
    """

    children: tuple[int, ...]
    exact: bool
    dtype: str
    bytes_read: int
    verify_seconds: float
    rebuild_seconds: float

    @property
    def total_seconds(self) -> float:
        """Verify plus rebuild time."""
        return self.verify_seconds + self.rebuild_seconds


def time_rebuild(
    directory: Path, indices: Sequence[int] | None = None, dtype: torch.dtype = torch.bfloat16
) -> RebuildTiming:
    """Rebuild from the shards in ``directory`` and time each phase.

    Args:
        directory: Directory written by :func:`~syncytium.shards.writer.write_children`.
        indices: Children to use; all of them by default.
        dtype: Dtype of the rebuilt weights; bf16, as main would load them, by default.

    Raises:
        ShardError: If ``indices`` repeats a child, or a shard or the rebuilt parent does not
            match the manifest.
    """
    manifest = Manifest.load(directory / MANIFEST_NAME)
    indices = tuple(range(manifest.scheme.num_children) if indices is None else indices)
    if len(set(indices)) != len(indices):
        raise ShardError(f"repeated child indices: {sorted(indices)}")
    shards = {index: directory / shard_name(index) for index in indices}
    start = time.perf_counter()
    for index, path in shards.items():
        verify_shard(path, manifest, index)
    verified = time.perf_counter()
    combine_shards(shards, manifest, dtype)
    done = time.perf_counter()
    return RebuildTiming(
        children=indices,
        exact=len(shards) == manifest.scheme.num_children,
        dtype=str(dtype).removeprefix("torch."),
        bytes_read=sum(path.stat().st_size for path in shards.values()),
        verify_seconds=verified - start,
        rebuild_seconds=done - verified,
    )


def log_timing(directory: Path, timing: RebuildTiming) -> None:
    """Append ``timing`` with the time and host to ``rebuilds.jsonl`` in ``directory``."""
    record = {
        "measured": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        **asdict(timing),
        "total_seconds": timing.total_seconds,
    }
    with (directory / LOG_NAME).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
