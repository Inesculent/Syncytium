"""Reading child shards and rebuilding weights from any set of them."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file
from torch import Tensor

from syncytium.quant import Grid, child_weights, parent_codes, parent_weights, partial_weights
from syncytium.quant.rotation import random_rotation
from syncytium.shards.manifest import (
    Manifest,
    ShardError,
    TensorSpec,
    file_sha256,
    update_codes_hash,
)
from syncytium.shards.packing import unpack


@dataclass(frozen=True)
class Child:
    """The contents of one verified shard.

    Attributes:
        index: Child index k.
        codes: Unpacked codes of each quantized tensor, in grouped shape.
        grids: Shared grid of each quantized tensor.
        full_precision: Unquantized tensors.
    """

    index: int
    codes: dict[str, Tensor]
    grids: dict[str, Grid]
    full_precision: dict[str, Tensor]


def read_child(path: Path, manifest: Manifest, index: int) -> Child:
    """Read child ``index`` after checking its file against the manifest.

    Raises:
        ShardError: If ``index`` is out of range or the file's hash does not match.
    """
    if not 0 <= index < manifest.scheme.num_children:
        raise ShardError(f"child index {index} is not in [0, {manifest.scheme.num_children})")
    if file_sha256(path) != manifest.shard_sha256[index]:
        raise ShardError(f"{path} does not match the manifest hash of child {index}")
    stored = load_file(path)
    child = Child(index, codes={}, grids={}, full_precision={})
    for spec in manifest.tensors:
        if spec.quantized:
            child.codes[spec.name], child.grids[spec.name] = _decode(spec, stored, manifest)
        else:
            child.full_precision[spec.name] = stored[spec.name]
    return child


def child_state_dict(child: Child, manifest: Manifest) -> dict[str, Tensor]:
    """Dequantized weights of a single child, ready for ``load_state_dict``."""
    scheme = manifest.scheme
    weights = {
        name: child_weights(
            codes, child.grids[name], child.index, scheme.num_children, scheme.phase
        )
        for name, codes in child.codes.items()
    }
    return _assemble(weights, child, manifest)


def rebuild_state_dict(children: Sequence[Child], manifest: Manifest) -> dict[str, Tensor]:
    """Weights rebuilt from any set of distinct children.

    With all N children the rebuild is exact and checked against the manifest's parent hash;
    with fewer it is the partial rebuild by interval intersection.

    Raises:
        ShardError: If ``children`` is empty, repeats an index, or a complete set does not
            reproduce the parent's hash.
    """
    _check_distinct(children)
    if len(children) == manifest.scheme.num_children:
        weights = _rebuild_exact(children, manifest)
    else:
        weights = _rebuild_partial(children, manifest)
    return _assemble(weights, children[0], manifest)


def _rebuild_exact(children: Sequence[Child], manifest: Manifest) -> dict[str, Tensor]:
    scheme, digest, weights = manifest.scheme, hashlib.sha256(), {}
    for spec in manifest.quantized():
        codes = parent_codes([child.codes[spec.name] for child in children])
        update_codes_hash(digest, spec.name, codes)
        grid = children[0].grids[spec.name]
        weights[spec.name] = parent_weights(codes, grid, scheme.num_children, scheme.phase)
    if digest.hexdigest() != manifest.parent_sha256:
        raise ShardError("rebuilt parent codes do not match the manifest hash")
    return weights


def _rebuild_partial(children: Sequence[Child], manifest: Manifest) -> dict[str, Tensor]:
    scheme = manifest.scheme
    return {
        spec.name: partial_weights(
            {child.index: child.codes[spec.name] for child in children},
            children[0].grids[spec.name],
            scheme.num_children,
            scheme.phase,
        )
        for spec in manifest.quantized()
    }


def _assemble(grouped: dict[str, Tensor], child: Child, manifest: Manifest) -> dict[str, Tensor]:
    """Rotate grouped weights back, restore their shapes and add unquantized tensors."""
    scheme, state = manifest.scheme, dict(child.full_precision)
    rotation = None
    if scheme.rotation_seed is not None:
        rotation = random_rotation(scheme.group_size, scheme.rotation_seed)
    for spec in manifest.quantized():
        weights = grouped[spec.name]
        if rotation is not None:
            weights = weights @ rotation.T
        state[spec.name] = weights.reshape(spec.shape)
    return state


def _decode(spec: TensorSpec, stored: dict[str, Tensor], manifest: Manifest) -> tuple[Tensor, Grid]:
    scheme = manifest.scheme
    codes = unpack(stored[f"{spec.name}/codes"], scheme.bits, spec.numel)
    grid = Grid(
        minimum=stored[f"{spec.name}/minimum"].float().unsqueeze(1),
        step=stored[f"{spec.name}/step"].float().unsqueeze(1),
        bits=scheme.bits,
    )
    return codes.reshape(-1, scheme.group_size), grid


def _check_distinct(children: Sequence[Child]) -> None:
    if not children:
        raise ShardError("at least one child is required")
    indices = [child.index for child in children]
    if len(set(indices)) != len(indices):
        raise ShardError(f"repeated child indices: {sorted(indices)}")
