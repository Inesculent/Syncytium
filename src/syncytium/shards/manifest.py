"""The manifest shared by a set of children: what they contain and how to verify them.

Every node keeps a copy. It records the scheme, the shape of every tensor, the SHA-256 of
each child's shard file and the SHA-256 of the parent's integer codes, so a rebuild can
prove it reproduced the parent bit for bit.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import Tensor

from syncytium.quant import Scheme

FORMAT = "syncytium-shards/1"
MANIFEST_NAME = "manifest.json"


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
    """Description of a complete set of N children."""

    model: str
    scheme: Scheme
    tensors: tuple[TensorSpec, ...]
    shard_sha256: tuple[str, ...]
    parent_sha256: str
    format: str = FORMAT

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
        )


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
    digest.update(codes.to("cpu").int().contiguous().numpy().astype("<i4").tobytes())
