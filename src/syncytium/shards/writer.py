"""Writing a model as N child shards plus a manifest.

Each shard is a safetensors file holding, for every quantized tensor, its packed codes and
the shared per-group minimum and step in fp16, plus every unquantized tensor unchanged.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor

from syncytium.quant import Grid, Scheme, fit_grid, parent_codes, quantize_children, to_groups
from syncytium.quant.rotation import random_rotation, random_signs
from syncytium.shards.manifest import (
    MANIFEST_NAME,
    Manifest,
    TensorSpec,
    file_sha256,
    shard_name,
    update_codes_hash,
)
from syncytium.shards.packing import pack

SCALE_DTYPE = torch.float16


def write_children(
    quantized: Mapping[str, Tensor],
    full_precision: Mapping[str, Tensor],
    scheme: Scheme,
    model: str,
    directory: Path,
) -> Manifest:
    """Quantize ``quantized`` into N children and write them with a manifest.

    Args:
        quantized: Weights to split into children, by name.
        full_precision: Tensors copied unchanged into every child, by name.
        scheme: Shared quantization settings.
        model: Identifier of the source model, recorded in the manifest.
        directory: Output directory; created if missing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    children: list[dict[str, Tensor]] = [{} for _ in range(scheme.num_children)]
    parent_hash = hashlib.sha256()
    for name, weight in quantized.items():
        grid, codes = quantize_tensor(weight, scheme)
        for child, child_codes in zip(children, codes, strict=True):
            child |= _encode(name, child_codes, grid)
        update_codes_hash(parent_hash, name, parent_codes(codes))
    shared = {name: t.detach().cpu().contiguous() for name, t in full_precision.items()}
    hashes = tuple(
        _save(directory / shard_name(index), tensors | shared)
        for index, tensors in enumerate(children)
    )
    seed = scheme.rotation_seed
    signs = None if seed is None else random_signs(scheme.group_size, seed)
    manifest = Manifest(
        model, scheme, _specs(quantized, shared), hashes, parent_hash.hexdigest(), signs
    )
    manifest.save(directory / MANIFEST_NAME)
    return manifest


def quantize_tensor(weight: Tensor, scheme: Scheme) -> tuple[Grid, list[Tensor]]:
    """Grid (exact in fp16) and every child's codes for one weight, in grouped shape."""
    groups = to_groups(weight.detach().to("cpu", torch.float32), scheme.group_size)
    if scheme.rotation_seed is not None:
        groups = groups @ random_rotation(scheme.group_size, scheme.rotation_seed)
    grid = fit_grid(groups, scheme.bits, storage=SCALE_DTYPE)
    return grid, quantize_children(groups, grid, scheme.num_children, scheme.phase)


def _encode(name: str, codes: Tensor, grid: Grid) -> dict[str, Tensor]:
    return {
        f"{name}/codes": pack(codes, grid.bits),
        f"{name}/minimum": grid.minimum.reshape(-1).to(SCALE_DTYPE),
        f"{name}/step": grid.step.reshape(-1).to(SCALE_DTYPE),
    }


def _specs(quantized: Mapping[str, Tensor], shared: Mapping[str, Tensor]) -> tuple[TensorSpec, ...]:
    return tuple(
        [TensorSpec(name, tuple(w.shape), quantized=True) for name, w in quantized.items()]
        + [TensorSpec(name, tuple(t.shape), quantized=False) for name, t in shared.items()]
    )


def _save(path: Path, tensors: dict[str, Tensor]) -> str:
    save_file(tensors, path)
    return file_sha256(path)
