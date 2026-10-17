"""The manifest shared by a set of children: what they contain and how to verify them.

Every node keeps a copy. It records the scheme, the shape of every tensor, the SHA-256 of
each child's shard file and the SHA-256 of the parent's integer codes, so a rebuild can
prove it reproduced the parent bit for bit. With a rotation it also stores the rotation's
signs: the parent hash covers codes in the rotated basis, so a reader that derived different
signs from the seed would pass the hash check and still return wrong weights.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import Tensor

from syncytium.quant import Scheme
from syncytium.quant.rotation import signed_hadamard

FORMAT = "syncytium-shards/1"
MANIFEST_NAME = "manifest.json"
# Codes hashed at once. Bounds the temporary int32 copy for large tensors.
HASH_CHUNK = 1 << 24


class ShardError(ValueError):
    """A shard or shard set is missing, corrupted or inconsistent with its manifest."""


@dataclass(frozen=True)
class TensorSpec:
    """Name, shape and storage of one model tensor."""

    name: str
    shape: tuple[int, ...]
    quantized: bool

    @property
    def numel(self) -> int:
        """Number of values in the tensor."""
        return math.prod(self.shape)


@dataclass(frozen=True)
class Manifest:
    """Description of a complete set of N children.

    Attributes:
        rotation_signs: Diagonal of D in the rotation R = D H / √g, or ``None`` without
            rotation. Readers use these rather than regenerating them from the seed.

    Raises:
        ShardError: If the scheme has a rotation and ``rotation_signs`` does not hold one
            sign per group element, or has signs without a rotation.
    """

    model: str
    scheme: Scheme
    tensors: tuple[TensorSpec, ...]
    shard_sha256: tuple[str, ...]
    parent_sha256: str
    rotation_signs: tuple[int, ...] | None = None
    format: str = FORMAT

    def __post_init__(self) -> None:
        if self.scheme.rotation_seed is None:
            if self.rotation_signs is not None:
                raise ShardError("manifest has rotation signs but no rotation")
        elif self.rotation_signs is None or len(self.rotation_signs) != self.scheme.group_size:
            raise ShardError(f"a rotated manifest needs {self.scheme.group_size} rotation signs")

    def rotation(self) -> Tensor | None:
        """The shared rotation R, or ``None`` without rotation."""
        if self.rotation_signs is None:
            return None
        return signed_hadamard(self.rotation_signs)

    def quantized(self) -> Iterator[TensorSpec]:
        """Specs of the quantized tensors, in hashing order."""
        return (spec for spec in self.tensors if spec.quantized)

    def save(self, path: Path) -> None:
        """Write the manifest as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Read a manifest written by :meth:`save`.

        Raises:
            ShardError: If the file is not a manifest of this format.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT:
            raise ShardError(f"{path} is not a {FORMAT} manifest")
        return cls(
            model=raw["model"],
            scheme=Scheme(**raw["scheme"]),
            tensors=tuple(
                TensorSpec(t["name"], tuple(t["shape"]), t["quantized"]) for t in raw["tensors"]
            ),
            shard_sha256=tuple(raw["shard_sha256"]),
            parent_sha256=raw["parent_sha256"],
            rotation_signs=_optional_tuple(raw.get("rotation_signs")),
        )


def _optional_tuple(values: list[int] | None) -> tuple[int, ...] | None:
    return None if values is None else tuple(values)


def shard_name(index: int) -> str:
    """File name of child ``index``'s shard."""
    return f"child-{index}.safetensors"


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def update_codes_hash(digest: "hashlib._Hash", name: str, codes: Tensor) -> None:
    """Add one tensor's parent codes (as little-endian int32) to a running hash."""
    digest.update(name.encode())
    for chunk in codes.reshape(-1).split(HASH_CHUNK):
        digest.update(chunk.to("cpu").int().contiguous().numpy().astype("<i4").tobytes())
