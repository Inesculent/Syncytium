"""Regrowth data next to a shard set, and regrowing lost shard files exactly (H14).

Two kinds of regrowth data, from ``docs/preregistration-swarm.md`` ("Self-healing"):

- *Neighbour bits* (any one loss): child i holds, for every quantized tensor, the bits that
  regrow child i - 1 (see :func:`~syncytium.quant.regrow.neighbour_bits`), packed one bit
  each. About 2/N bits per weight per child.
- *Coded residue* (any t losses): the residues r = Q mod N, packed to ⌈log₂ N⌉ bits, are cut
  into N - t blocks and spread over N shares by a Reed-Solomon code
  (:mod:`syncytium.shards.erasure`); child i holds share i. About ⌈log₂ N⌉/(N - t) bits per
  weight per child.

Each child's regrowth data is one safetensors file next to its shard, and a sidecar JSON
records the hash of every such file and the parent hash of the set it belongs to. The shard
files and the manifest are left untouched, and nothing here is read for serving. A regrown
shard is written exactly as :func:`~syncytium.shards.writer.write_children` wrote it, so it
must match its SHA-256 in the manifest; a regrowth is refused before writing if the parent
codes it implies do not match the parent hash.
"""

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from syncytium.quant import (
    ambiguous,
    neighbour_bits,
    parent_codes,
    regrow_from_residues,
    regrow_with_neighbour_bits,
    residues,
)
from syncytium.shards.erasure import ORDER, decode, encode
from syncytium.shards.manifest import (
    MANIFEST_NAME,
    Manifest,
    ShardError,
    TensorSpec,
    file_sha256,
    shard_name,
    update_codes_hash,
)
from syncytium.shards.packing import pack, packed_size, unpack
from syncytium.shards.reader import verify_shard

REGROWTH_FORMAT = "syncytium-regrowth/1"
# The shared grid is identical in every child, so a survivor's copy is the lost child's.
GRID_PARTS = ("minimum", "step")


class Redundancy(StrEnum):
    """Which regrowth data a shard set stores."""

    NEIGHBOUR = "neighbour"  # any one loss, or any losses with no two adjacent children
    RESIDUE = "residue"  # any t losses


@dataclass(frozen=True)
class RegrowthManifest:
    """The sidecar describing one kind of regrowth data for a shard set.

    Attributes:
        kind: Which regrowth data the files hold.
        tolerated: Losses survived at once: 1 for neighbour bits, t for the coded residue.
        parent_sha256: Parent hash of the shard set the data belongs to.
        file_sha256: SHA-256 of each child's regrowth file, by child index.
    """

    kind: Redundancy
    tolerated: int
    parent_sha256: str
    file_sha256: tuple[str, ...]
    format: str = REGROWTH_FORMAT

    def save(self, path: Path) -> None:
        """Write the sidecar as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RegrowthManifest":
        """Read a sidecar written by :meth:`save`.

        Raises:
            ShardError: If the file is not a sidecar of this format.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != REGROWTH_FORMAT:
            raise ShardError(f"{path} is not a {REGROWTH_FORMAT} sidecar")
        return cls(
            kind=Redundancy(raw["kind"]),
            tolerated=raw["tolerated"],
            parent_sha256=raw["parent_sha256"],
            file_sha256=tuple(raw["file_sha256"]),
        )


def regrowth_name(kind: Redundancy, index: int) -> str:
    """File name of the regrowth data held by child ``index``."""
    return f"regrow-{kind}-{index}.safetensors"


def regrowth_manifest_name(kind: Redundancy) -> str:
    """File name of the sidecar for ``kind``."""
    return f"regrow-{kind}.json"


def residue_width(num_children: int) -> int:
    """Bits per packed residue, ⌈log₂ N⌉ (at least one)."""
    return max(1, (num_children - 1).bit_length())


def check_tolerated(kind: Redundancy, num_children: int, tolerated: int) -> None:
    """Check that ``kind`` can be built for N children surviving ``tolerated`` losses.

    Raises:
        ShardError: If N < 2, the neighbour bits are asked to survive other than one loss,
            or the coded residue has t outside [1, N - 1] or N > 255.
    """
    if num_children < 2:
        raise ShardError(f"regrowth needs at least two children, got {num_children}")
    if kind is Redundancy.NEIGHBOUR and tolerated != 1:
        raise ShardError(f"neighbour bits survive exactly one loss, not {tolerated}")
    if kind is Redundancy.RESIDUE and not 1 <= tolerated < num_children <= ORDER:
        raise ShardError(
            f"the coded residue needs 1 ≤ t < N ≤ {ORDER}, got t = {tolerated}, N = {num_children}"
        )


def check_losses(
    lost: Collection[int], num_children: int, kind: Redundancy, tolerated: int
) -> None:
    """Check that ``kind`` can regrow the children in ``lost``.

    Raises:
        ShardError: If an index is out of range; for neighbour bits, if two lost children
            are adjacent (child j + 1 holds child j's bits and bounds it from above); for the
            coded residue, if more than t children are lost.
    """
    if not all(0 <= index < num_children for index in lost):
        raise ShardError(f"lost children {sorted(lost)} are not all in [0, {num_children})")
    if kind is Redundancy.RESIDUE and len(lost) > tolerated:
        raise ShardError(
            f"{len(lost)} children lost but the coded residue survives at most {tolerated}"
        )
    if kind is Redundancy.NEIGHBOUR:
        for index in sorted(lost):
            after = (index + 1) % num_children
            if after in lost:
                raise ShardError(
                    f"children {index} and {after} are both lost: child {after} holds the "
                    f"neighbour bits of child {index}, so neither can be regrown exactly; "
                    "neighbour bits survive one loss (use the coded residue for more)"
                )


def regrowth_data(children: Sequence[Tensor], kind: Redundancy, tolerated: int = 1) -> list[Tensor]:
    """The regrowth data each child holds for one tensor, by child index.

    Args:
        children: Codes of all N children of the tensor.
        kind: Which regrowth data to build.
        tolerated: Losses to survive; t for the coded residue, 1 for neighbour bits.

    Returns:
        One 1-D ``uint8`` tensor per child.

    Raises:
        ShardError: If ``kind`` cannot survive ``tolerated`` losses of N children.
        ValueError: If the children are not staggered.
    """
    num_children = len(children)
    check_tolerated(kind, num_children, tolerated)
    if kind is Redundancy.RESIDUE:
        return list(residue_shares(children, tolerated))
    # Child i holds the bits of child i - 1.
    held = [(index - 1) % num_children for index in range(num_children)]
    return [pack(neighbour_bits(children, lost), 1) for lost in held]


def regrow_codes(
    available: Mapping[int, Tensor],
    held: Mapping[int, Tensor],
    lost: Collection[int],
    kind: Redundancy,
    tolerated: int = 1,
) -> dict[int, Tensor]:
    """Regrow the lost children of one tensor exactly from the survivors.

    Args:
        available: Codes of every surviving child, keyed by child index.
        held: Regrowth data held by the surviving children (from :func:`regrowth_data`).
        lost: Indices of the lost children; together with ``available`` they make N.
        kind: Which regrowth data ``held`` is.
        tolerated: Losses the data was built to survive.

    Returns:
        Codes of each lost child, shaped like the survivors' codes.

    Raises:
        ShardError: If ``kind`` cannot regrow this loss pattern.
    """
    num_children = len(available) + len(lost)
    check_losses(lost, num_children, kind, tolerated)
    if kind is Redundancy.RESIDUE:
        return _regrow_from_shares(available, held, lost, tolerated)
    return {index: _regrow_from_bits(available, held, index, num_children) for index in lost}


def residue_shares(children: Sequence[Tensor], tolerated: int) -> Tensor:
    """Pack one tensor's residues and spread them over N shares; any N - t recover them.

    Returns:
        ``uint8`` tensor of shape ``(N, length)``; row i is held by child i.
    """
    num_children = len(children)
    packed = pack(residues(children), residue_width(num_children))
    blocks = num_children - tolerated
    length = -(-packed.numel() // blocks)
    data = torch.nn.functional.pad(packed, (0, blocks * length - packed.numel()))
    return encode(data.reshape(blocks, length), num_children)


def decode_residues(
    shares: Mapping[int, Tensor], count: int, num_children: int, tolerated: int
) -> Tensor:
    """Recover ``count`` residues (flat) from any N - t of the shares of :func:`residue_shares`.

    Raises:
        ValueError: If fewer than N - t shares are given.
    """
    data = decode(shares, num_children - tolerated, num_children).reshape(-1)
    return unpack(data, residue_width(num_children), count)


def write_regrowth(directory: Path, kind: Redundancy, tolerated: int = 1) -> RegrowthManifest:
    """Write regrowth data next to the shard set in ``directory``, with its sidecar.

    Every shard is checked against the manifest first. The shard files and manifest are not
    changed.

    Args:
        directory: Directory written by :func:`~syncytium.shards.writer.write_children`.
        kind: Which regrowth data to write.
        tolerated: Losses to survive; t for the coded residue, 1 for neighbour bits.

    Raises:
        ShardError: If a shard does not match the manifest, or ``kind`` cannot survive
            ``tolerated`` losses of N children.
    """
    manifest = Manifest.load(directory / MANIFEST_NAME)
    num_children = manifest.scheme.num_children
    check_tolerated(kind, num_children, tolerated)
    hashes = _write_regrowth_files(directory, manifest, kind, tolerated, range(num_children))
    sidecar = RegrowthManifest(
        kind, tolerated, manifest.parent_sha256, tuple(hashes[k] for k in range(num_children))
    )
    sidecar.save(directory / regrowth_manifest_name(kind))
    return sidecar


def regrow_shards(directory: Path, lost: Collection[int], kind: Redundancy) -> list[Path]:
    """Regrow the shard files (and regrowth files) of the ``lost`` children in ``directory``.

    The survivors' shards and regrowth files are checked against their hashes, the lost
    children's codes are regrown, and the parent codes they imply are checked against the
    parent hash before anything is written. Each regrown shard and regrowth file must then
    match its recorded SHA-256, so the set returns to full health bit for bit.

    Args:
        directory: Directory holding the shard set and ``kind``'s regrowth data.
        lost: Indices of the lost children. Their files may be missing or corrupt; they are
            never read.
        kind: Which regrowth data to use.

    Returns:
        Paths of the regrown shard files.

    Raises:
        ShardError: If ``kind`` cannot regrow this loss pattern, a survivor's file or the
            sidecar does not match, or the regrown data does not reproduce the hashes.
    """
    manifest = Manifest.load(directory / MANIFEST_NAME)
    sidecar = _load_sidecar(directory, kind, manifest)
    num_children = manifest.scheme.num_children
    lost = sorted(set(lost))
    check_losses(lost, num_children, kind, sidecar.tolerated)
    survivors = [index for index in range(num_children) if index not in lost]
    for index in survivors:
        verify_shard(directory / shard_name(index), manifest, index)
        _verify(directory / regrowth_name(kind, index), sidecar.file_sha256[index])
    shards = _regrow_shard_tensors(directory, manifest, sidecar, lost)
    paths = []
    for index, tensors in shards.items():
        path = directory / shard_name(index)
        _save_verified(path, tensors, manifest.shard_sha256[index])
        paths.append(path)
    hashes = _write_regrowth_files(directory, manifest, kind, sidecar.tolerated, lost)
    for index, digest in hashes.items():
        if digest != sidecar.file_sha256[index]:
            raise ShardError(f"regrown {regrowth_name(kind, index)} does not match the sidecar")
    return paths


@dataclass(frozen=True)
class _OpenSet:
    """Open survivor files of a shard set, read one tensor at a time."""

    manifest: Manifest
    sidecar: RegrowthManifest
    shards: dict[int, Any]
    held: dict[int, Any]

    def codes(self, spec: TensorSpec) -> dict[int, Tensor]:
        return {k: _read_codes(spec, f, self.manifest) for k, f in self.shards.items()}

    def regrowth(self, spec: TensorSpec) -> dict[int, Tensor]:
        return {k: f.get_tensor(spec.name) for k, f in self.held.items()}

    def first(self, name: str) -> Tensor:
        return next(iter(self.shards.values())).get_tensor(name)


def _regrow_shard_tensors(
    directory: Path, manifest: Manifest, sidecar: RegrowthManifest, lost: Sequence[int]
) -> dict[int, dict[str, Tensor]]:
    """Every tensor of each lost child's shard, checked against the parent hash."""
    regrown: dict[int, dict[str, Tensor]] = {index: {} for index in lost}
    digest = hashlib.sha256()
    with ExitStack() as stack:
        survivors = [k for k in range(manifest.scheme.num_children) if k not in regrown]
        source = _OpenSet(
            manifest,
            sidecar,
            shards={k: _open(stack, directory / shard_name(k)) for k in survivors},
            held={k: _open(stack, directory / regrowth_name(sidecar.kind, k)) for k in survivors},
        )
        for spec in manifest.tensors:
            for index, tensors in _regrow_tensor(spec, source, lost, digest).items():
                regrown[index] |= tensors
    if digest.hexdigest() != manifest.parent_sha256:
        raise ShardError("regrown children do not reproduce the parent hash")
    return regrown


def _regrow_tensor(
    spec: TensorSpec, source: _OpenSet, lost: Sequence[int], digest: "hashlib._Hash"
) -> dict[int, dict[str, Tensor]]:
    """One tensor's entries in each lost shard; quantized ones also extend the parent hash."""
    if not spec.quantized:
        return {index: {spec.name: source.first(spec.name)} for index in lost}
    available = source.codes(spec)
    codes = regrow_codes(
        available, source.regrowth(spec), lost, source.sidecar.kind, source.sidecar.tolerated
    )
    update_codes_hash(digest, spec.name, parent_codes([*available.values(), *codes.values()]))
    grid = {f"{spec.name}/{part}": source.first(f"{spec.name}/{part}") for part in GRID_PARTS}
    bits = source.manifest.scheme.bits
    return {index: {f"{spec.name}/codes": pack(c, bits), **grid} for index, c in codes.items()}


def _write_regrowth_files(
    directory: Path,
    manifest: Manifest,
    kind: Redundancy,
    tolerated: int,
    holders: Collection[int],
) -> dict[int, str]:
    """Build the regrowth files of ``holders`` from the full (verified) shard set."""
    num_children = manifest.scheme.num_children
    for index in range(num_children):
        verify_shard(directory / shard_name(index), manifest, index)
    payloads: dict[int, dict[str, Tensor]] = {index: {} for index in holders}
    with ExitStack() as stack:
        files = [_open(stack, directory / shard_name(k)) for k in range(num_children)]
        for spec in manifest.quantized():
            children = [_read_codes(spec, file, manifest) for file in files]
            data = regrowth_data(children, kind, tolerated)
            for index, tensors in payloads.items():
                tensors[spec.name] = data[index]
    return {
        index: _save(directory / regrowth_name(kind, index), tensors)
        for index, tensors in payloads.items()
    }


def _regrow_from_bits(
    available: Mapping[int, Tensor], held: Mapping[int, Tensor], index: int, num_children: int
) -> Tensor:
    reference = next(iter(available.values()))
    count = int(ambiguous(available, index, num_children).sum())
    packed = held[(index + 1) % num_children]
    if packed.numel() != packed_size(count, 1):
        raise ShardError(f"child {index} needs {count} neighbour bits; the stored data differs")
    bits = unpack(packed, 1, count)
    return regrow_with_neighbour_bits(available, index, bits, num_children).reshape(reference.shape)


def _regrow_from_shares(
    available: Mapping[int, Tensor],
    held: Mapping[int, Tensor],
    lost: Collection[int],
    tolerated: int,
) -> dict[int, Tensor]:
    num_children = len(available) + len(lost)
    # Any survivor works; m = ⌊Q/N⌋ follows from its codes and the residues.
    survivor, codes = next(iter(sorted(available.items())))
    residue = decode_residues(held, codes.numel(), num_children, tolerated).reshape(codes.shape)
    return {k: regrow_from_residues(residue, survivor, codes, k, num_children) for k in lost}


def _load_sidecar(directory: Path, kind: Redundancy, manifest: Manifest) -> RegrowthManifest:
    sidecar = RegrowthManifest.load(directory / regrowth_manifest_name(kind))
    if sidecar.kind is not kind or sidecar.parent_sha256 != manifest.parent_sha256:
        raise ShardError(f"{regrowth_manifest_name(kind)} belongs to a different shard set")
    if len(sidecar.file_sha256) != manifest.scheme.num_children:
        raise ShardError(f"{regrowth_manifest_name(kind)} does not list one file per child")
    return sidecar


def _read_codes(spec: TensorSpec, file: Any, manifest: Manifest) -> Tensor:
    scheme = manifest.scheme
    codes = unpack(file.get_tensor(f"{spec.name}/codes"), scheme.bits, spec.numel)
    return codes.reshape(-1, scheme.group_size)


def _open(stack: ExitStack, path: Path) -> Any:
    return stack.enter_context(safe_open(str(path), "pt"))


def _verify(path: Path, expected: str) -> None:
    if not path.exists() or file_sha256(path) != expected:
        raise ShardError(f"{path} does not match its hash in the regrowth sidecar")


def _save(path: Path, tensors: dict[str, Tensor]) -> str:
    save_file(tensors, path)
    return file_sha256(path)


def _save_verified(path: Path, tensors: dict[str, Tensor], expected: str) -> None:
    if _save(path, tensors) != expected:
        path.unlink()
        raise ShardError(f"regrown {path.name} does not match the manifest hash")
