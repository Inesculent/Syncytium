"""Reading child shards and rebuilding weights from any set of them.

A rebuild streams the shards: tensors are read one at a time from every shard, and each
tensor's codes are combined and dequantized before the next is read. Peak memory is the
rebuilt weights plus a few copies of the largest tensor, not N whole shards.

The channel scales of a scaled set are read from the scales file next to the shards, after
checking it against the manifest. Dequantization rotates back first and divides by the scales
after, undoing the writer's transforms in reverse order (``docs/method.md``, "Optional channel
scaling").
"""

import hashlib
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor

from syncytium.quant import Grid, child_weights, parent_codes, parent_weights, partial_weights
from syncytium.quant.simulate import CHUNK_GROUPS
from syncytium.shards.manifest import (
    SCALES_NAME,
    Manifest,
    ShardError,
    TensorSpec,
    file_sha256,
    update_parent_hash,
)
from syncytium.shards.packing import unpack

# Reads one tensor of a shard file by name (from ``load_file`` or ``safe_open``).
Getter = Callable[[str], Tensor]
# Dequantizes a range of groups, given that range and the grid restricted to it.
Estimate = Callable[[slice, Grid], Tensor]


@dataclass(frozen=True)
class Child:
    """The contents of one verified shard.

    Attributes:
        index: Child index k.
        codes: Unpacked codes of each quantized tensor, in grouped shape.
        grids: Shared grid of each quantized tensor.
        full_precision: Unquantized tensors.
        scales: Channel scales of each scaled tensor, shared by every child; empty for an
            unscaled set.
    """

    index: int
    codes: dict[str, Tensor]
    grids: dict[str, Grid]
    full_precision: dict[str, Tensor]
    scales: dict[str, Tensor] = field(default_factory=dict)


def verify_shard(path: Path, manifest: Manifest, index: int) -> None:
    """Check that ``path`` is child ``index`` of the set described by ``manifest``.

    Raises:
        ShardError: If ``index`` is out of range or the file's hash does not match.
    """
    if not 0 <= index < manifest.scheme.num_children:
        raise ShardError(f"child index {index} is not in [0, {manifest.scheme.num_children})")
    if file_sha256(path) != manifest.shard_sha256[index]:
        raise ShardError(f"{path} does not match the manifest hash of child {index}")


def read_scales(path: Path, manifest: Manifest) -> dict[str, Tensor]:
    """Channel scales of each scaled tensor, after checking their file against the manifest.

    An unscaled set has no scales file, and ``path`` is not read.

    Returns:
        One fp32 vector per scaled tensor, by name; empty for an unscaled set.

    Raises:
        ShardError: If the set is scaled and the file is missing or does not match its hash.
    """
    if manifest.scales_sha256 is None:
        return {}
    if not path.exists() or file_sha256(path) != manifest.scales_sha256:
        raise ShardError(f"{path} does not match the manifest hash of the channel scales")
    return load_file(path)


def read_child(path: Path, manifest: Manifest, index: int) -> Child:
    """Read child ``index`` after checking its file against the manifest.

    A scaled set's scales are read from :data:`~syncytium.shards.manifest.SCALES_NAME` next
    to the shard, and checked against the manifest too.

    Raises:
        ShardError: If ``index`` is out of range, or the shard's or scales' hash does not match.
    """
    verify_shard(path, manifest, index)
    scales = read_scales(path.with_name(SCALES_NAME), manifest)
    stored = load_file(path)
    child = Child(index, codes={}, grids={}, full_precision={}, scales=scales)
    for spec in manifest.tensors:
        if spec.quantized:
            child.codes[spec.name] = _codes(spec, stored.__getitem__, manifest)
            child.grids[spec.name] = _grid(spec, stored.__getitem__, manifest)
        else:
            child.full_precision[spec.name] = stored[spec.name]
    return child


def child_state_dict(
    child: Child, manifest: Manifest, dtype: torch.dtype = torch.float32
) -> dict[str, Tensor]:
    """Dequantized weights of a single child, ready for ``load_state_dict``."""
    state = dict(child.full_precision)
    for spec in manifest.quantized():
        estimate = _child_estimate(child.codes[spec.name], child.index, manifest)
        grid, scales = child.grids[spec.name], child.scales.get(spec.name)
        state[spec.name] = _dequantize(spec, grid, estimate, manifest, dtype, scales)
    return state


def rebuild_from_files(
    shards: Mapping[int, Path], manifest: Manifest, dtype: torch.dtype = torch.float32
) -> dict[str, Tensor]:
    """Weights rebuilt from the shard files of any set of children.

    Every file is checked against the manifest before any is used. With all N children the
    rebuild is exact and checked against the parent hash; with fewer it is the partial
    rebuild by interval intersection. A scaled set's scales are read from the scales file
    next to the shard of the lowest child index.

    Args:
        shards: Path of each available child's shard, keyed by child index k.
        manifest: The set's manifest.
        dtype: Dtype of the rebuilt weights. They are computed in float32 and cast per chunk.

    Raises:
        ShardError: If ``shards`` is empty, a file (shard or scales) does not match its hash,
            or a complete set does not reproduce the parent hash.
    """
    if not shards:
        raise ShardError("at least one child is required")
    for index, path in shards.items():
        verify_shard(path, manifest, index)
    return combine_shards(shards, manifest, dtype)


def combine_shards(
    shards: Mapping[int, Path], manifest: Manifest, dtype: torch.dtype = torch.float32
) -> dict[str, Tensor]:
    """The combining step of :func:`rebuild_from_files`, for files already verified.

    Only call this on shards that passed :func:`verify_shard`; it is separate so the two steps
    can be timed apart. A scaled set's scales file is checked against its hash here, and a
    complete set against the parent hash, which covers the scales too.

    Raises:
        ShardError: If the scales file does not match its hash, or a complete set does not
            reproduce the parent hash.
    """
    scheme, digest = manifest.scheme, hashlib.sha256()
    exact = len(shards) == scheme.num_children
    scales = read_scales(shards[min(shards)].with_name(SCALES_NAME), manifest)
    with ExitStack() as stack:
        files = {k: stack.enter_context(safe_open(str(p), "pt")) for k, p in sorted(shards.items())}
        first = next(iter(files.values()))
        state = {s.name: first.get_tensor(s.name) for s in manifest.tensors if not s.quantized}
        for spec in manifest.quantized():
            if exact:
                codes = parent_codes(_codes(spec, f.get_tensor, manifest) for f in files.values())
                update_parent_hash(digest, spec.name, codes, scales.get(spec.name))
                estimate = _parent_estimate(codes, manifest)
            else:
                available = {k: _codes(spec, f.get_tensor, manifest) for k, f in files.items()}
                estimate = _partial_estimate(available, manifest)
            grid = _grid(spec, first.get_tensor, manifest)
            state[spec.name] = _dequantize(
                spec, grid, estimate, manifest, dtype, scales.get(spec.name)
            )
    if exact and digest.hexdigest() != manifest.parent_sha256:
        raise ShardError("rebuilt parent codes do not match the manifest hash")
    return state


def _child_estimate(codes: Tensor, index: int, manifest: Manifest) -> Estimate:
    n, phase = manifest.scheme.num_children, manifest.scheme.phase
    return lambda rows, grid: child_weights(codes[rows], grid, index, n, phase)


def _parent_estimate(codes: Tensor, manifest: Manifest) -> Estimate:
    n, phase = manifest.scheme.num_children, manifest.scheme.phase
    return lambda rows, grid: parent_weights(codes[rows], grid, n, phase)


def _partial_estimate(available: Mapping[int, Tensor], manifest: Manifest) -> Estimate:
    n, phase = manifest.scheme.num_children, manifest.scheme.phase
    return lambda rows, grid: partial_weights(
        {k: codes[rows] for k, codes in available.items()}, grid, n, phase
    )


def _dequantize(
    spec: TensorSpec,
    grid: Grid,
    estimate: Estimate,
    manifest: Manifest,
    dtype: torch.dtype,
    scales: Tensor | None = None,
) -> Tensor:
    """Run ``estimate`` over the groups in chunks, rotate back, unscale and restore the shape.

    Each chunk is rotated back and divided by its scales in float32 before it is cast.
    """
    rotation = manifest.rotation()
    groups, size = grid.minimum.shape[0], manifest.scheme.group_size
    weights = torch.empty(groups, size, dtype=dtype)
    for start in range(0, groups, CHUNK_GROUPS):
        rows = slice(start, start + CHUNK_GROUPS)
        chunk = estimate(rows, Grid(grid.minimum[rows], grid.step[rows], grid.bits))
        if rotation is not None:
            chunk = chunk @ rotation.T
        if scales is not None:
            chunk = chunk / _group_scales(scales, start, chunk.shape[0], size)
        weights[rows] = chunk
    return weights.reshape(spec.shape)


def _group_scales(scales: Tensor, start: int, count: int, group_size: int) -> Tensor:
    """Scales of ``count`` consecutive groups from group ``start``, in grouped shape.

    Groups run along the input dimension, so each row of the weight holds len(scales) / g
    groups, and group i uses the (i mod len(scales) / g)-th run of g scales.
    """
    per_row = scales.reshape(-1, group_size)
    return per_row[torch.arange(start, start + count) % per_row.shape[0]]


def _codes(spec: TensorSpec, get: Getter, manifest: Manifest) -> Tensor:
    scheme = manifest.scheme
    codes = unpack(get(f"{spec.name}/codes"), scheme.bits, spec.numel)
    return codes.reshape(-1, scheme.group_size)


def _grid(spec: TensorSpec, get: Getter, manifest: Manifest) -> Grid:
    return Grid(
        minimum=get(f"{spec.name}/minimum").float().unsqueeze(1),
        step=get(f"{spec.name}/step").float().unsqueeze(1),
        bits=manifest.scheme.bits,
    )
