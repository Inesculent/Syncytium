"""A member's model built from its own shard alone, holding packed codes (H18).

Every quantized linear and embedding layer keeps only its child's packed codes, the shared
grid and the shared channel scales, and dequantizes a block of rows at a time when it is used.
The full weights never exist on the device: a 3-bit 9B child takes 3.9 GB, against 18 GB for
the bf16 parent. The rest of the model (norms, convolutions, decay parameters) is read from the
shard's full-precision tensors.

When a layer's :attr:`PackedWeight.exchange` is set, its output is averaged with the other live
members' before anything reads it, which makes the group compute the fused model.
"""

import copy
import functools
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn.functional import linear
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from syncytium.models import quantized_weights, split_parameters
from syncytium.quant import Grid, child_weights
from syncytium.shards import MANIFEST_NAME, Manifest, read_scales, shard_name, verify_shard
from syncytium.shards.manifest import SCALES_NAME, TensorSpec

# Weights dequantized at once; bounds the temporary float32 block.
BLOCK = 1 << 24

Average = Callable[[Tensor], Tensor]

_LAYER = re.compile(r"^(model\.layers\.)(\d+)(\..*)$")


def unpack_fast(packed: Tensor, bits: int, start: int, count: int) -> Tensor:
    """Codes ``start`` to ``start + count`` of a packed stream, on the stream's device.

    Every run of 8 codes fills exactly ``bits`` bytes (see :mod:`syncytium.shards.packing`),
    so with ``start`` and ``count`` multiples of 8 the run is read as whole ``bits``-byte words.
    """
    if start % 8 or count % 8:
        raise ValueError(f"start and count must be multiples of 8, got {start} and {count}")
    words = packed[start * bits // 8 : (start + count) * bits // 8].reshape(-1, bits).long()
    value = torch.zeros(words.shape[0], dtype=torch.long, device=packed.device)
    for byte in range(bits):
        value |= words[:, byte] << (8 * byte)
    shifts = torch.arange(8, device=packed.device) * bits
    return ((value.unsqueeze(1) >> shifts) & ((1 << bits) - 1)).reshape(-1).to(torch.uint8)


class PackedWeight(nn.Module):
    """One quantized weight of one child, kept packed on the device.

    Args:
        spec: The weight's name and shape.
        stored: Its ``codes``, ``minimum`` and ``step`` tensors, as in the shard file.
        manifest: The shard set's manifest (scheme and rotation).
        index: Which child this is.
        scales: The shared channel scales, or ``None``.
    """

    def __init__(
        self,
        spec: TensorSpec,
        stored: Mapping[str, Tensor],
        manifest: Manifest,
        index: int,
        scales: Tensor | None,
    ) -> None:
        super().__init__()
        device = stored["codes"].device
        self.shape, self.index, self.scheme = spec.shape, index, manifest.scheme
        self.register_buffer("packed", stored["codes"], persistent=False)
        self.register_buffer("minimum", stored["minimum"], persistent=False)
        self.register_buffer("step", stored["step"], persistent=False)
        rotation = manifest.rotation()
        self.register_buffer(
            "rotation", None if rotation is None else rotation.to(device), persistent=False
        )
        self.register_buffer(
            "scales", None if scales is None else scales.to(device), persistent=False
        )
        self.exchange: Average | None = None

    @property
    def nbytes(self) -> int:
        """Device memory this weight occupies."""
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def rows(self, first: int, last: int, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        """Dequantized rows ``first`` to ``last`` of the weight."""
        width, size = self.shape[-1], self.scheme.group_size
        codes = unpack_fast(self.packed, self.scheme.bits, first * width, (last - first) * width)
        groups = slice(first * width // size, last * width // size)
        block = self._dequantize(codes, self.minimum[groups], self.step[groups], last - first)
        return block.to(dtype)

    def select(self, rows: Tensor, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        """Dequantized ``rows`` of the weight (any order)."""
        width, size, bits = self.shape[-1], self.scheme.group_size, self.scheme.bits
        chosen = self.packed.reshape(self.shape[0], width * bits // 8)[rows].reshape(-1)
        codes = unpack_fast(chosen, bits, 0, rows.numel() * width)
        per_row = width // size
        minimum = self.minimum.reshape(-1, per_row)[rows].reshape(-1)
        step = self.step.reshape(-1, per_row)[rows].reshape(-1)
        return self._dequantize(codes, minimum, step, rows.numel()).to(dtype)

    def blocks(self) -> list[tuple[int, int]]:
        """Row ranges small enough to dequantize at once."""
        rows, width = self.shape[0], self.shape[-1]
        step = max(1, BLOCK // width)
        return [(first, min(first + step, rows)) for first in range(0, rows, step)]

    def share(self, output: Tensor) -> Tensor:
        """Average ``output`` with the group's when fused; unchanged when alone."""
        return output if self.exchange is None else self.exchange(output)

    def _dequantize(self, codes: Tensor, minimum: Tensor, step: Tensor, rows: int) -> Tensor:
        size, n, phase = self.scheme.group_size, self.scheme.num_children, self.scheme.phase
        grid = Grid(minimum.float().unsqueeze(1), step.float().unsqueeze(1), self.scheme.bits)
        block = child_weights(codes.reshape(-1, size), grid, self.index, n, phase)
        if self.rotation is not None:
            block = block @ self.rotation.T
        block = block.reshape(rows, self.shape[-1])
        if self.scales is not None:
            block = block / self.scales
        return block


class PackedLinear(nn.Module):
    """A linear layer computed block by block from a :class:`PackedWeight`."""

    def __init__(self, weight: PackedWeight, has_bias: bool) -> None:
        super().__init__()
        self.packed_weight = weight
        self.in_features, self.out_features = weight.shape[-1], weight.shape[0]
        bias = torch.empty(self.out_features, device="meta") if has_bias else None
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """``x W^T + b``, then shared with the group if fused."""
        w = self.packed_weight
        output = torch.cat([linear(x, w.rows(a, b, x.dtype)) for a, b in w.blocks()], dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return w.share(output)


class PackedEmbedding(nn.Module):
    """An embedding that dequantizes only the rows it looks up."""

    def __init__(self, weight: PackedWeight, padding_idx: int | None) -> None:
        super().__init__()
        self.packed_weight = weight
        self.num_embeddings, self.embedding_dim = weight.shape
        self.padding_idx = padding_idx
        self.dtype = torch.bfloat16  # set to the model's dtype when it is loaded

    def forward(self, ids: Tensor) -> Tensor:
        """The rows of ``ids``, then shared with the group if fused."""
        w = self.packed_weight
        unique, inverse = torch.unique(ids, return_inverse=True)
        return w.share(w.select(unique, self.dtype)[inverse])


@dataclass(frozen=True)
class Part:
    """Which part of a model to build: child ``index``'s layers ``first`` to ``last``.

    ``last`` of ``None`` means every layer. A part that stops before the last layer ends with
    the hidden state (no final norm or head); one that starts after the first layer takes a
    hidden state instead of tokens, and keeps the embedding only when the head is tied to it.
    """

    index: int
    first: int = 0
    last: int | None = None


def load_member(
    directory: Path, index: int, device: str = "cuda", config: PretrainedConfig | None = None
) -> PreTrainedModel:
    """Build child ``index``'s model from its shard in ``directory``, without the full weights.

    ``config`` defaults to the configuration of the model named in the manifest.

    Raises:
        ShardError: If the shard or the scales file does not match the manifest.
    """
    manifest = Manifest.load(directory / MANIFEST_NAME)
    path = directory / shard_name(index)
    verify_shard(path, manifest, index)
    scales = read_scales(directory / SCALES_NAME, manifest)
    return build_model(path, manifest, scales, Part(index), device, config)


def build_model(
    path: Path,
    manifest: Manifest,
    scales: Mapping[str, Tensor],
    part: Part,
    device: str = "cuda",
    config: PretrainedConfig | None = None,
) -> PreTrainedModel:
    """Build ``part`` of a child's model from the tensors in ``path``, holding packed codes.

    ``path`` needs only the tensors the part reads (see :func:`part_names`), under the full
    model's names.
    """
    model = part_skeleton(config or AutoConfig.from_pretrained(manifest.model), part)
    rename = functools.partial(global_name, first=part.first)
    names = {rename(n) for n in quantized_weights(model)}
    with safe_open(str(path), "pt", device=device) as file:
        packed = {
            spec.name: PackedWeight(
                spec,
                {
                    piece: file.get_tensor(f"{spec.name}/{piece}")
                    for piece in ("codes", "minimum", "step")
                },
                manifest,
                part.index,
                scales.get(spec.name),
            )
            for spec in manifest.quantized()
            if spec.name in names
        }
        _swap_layers(model, packed, rename)
        for name, _ in list(model.named_parameters()):
            _set_parameter(model, name, file.get_tensor(rename(name)))
    _reset_rotary(model, device)
    dtype = next(model.parameters()).dtype
    for module in model.modules():
        if isinstance(module, PackedEmbedding):
            module.dtype = dtype
    left = [n for n, t in chain(model.named_parameters(), model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"tensors not loaded from the shard: {left[:5]}")
    return model.eval().requires_grad_(False)


def part_skeleton(config: PretrainedConfig, part: Part) -> PreTrainedModel:
    """The model of ``part`` on the meta device, its layers numbered from 0."""
    total = config.num_hidden_layers
    last = total if part.last is None else part.last
    if (part.first, last) != (0, total):
        config = copy.deepcopy(config)
        config.layer_types = config.layer_types[part.first : last]
        config.num_hidden_layers = last - part.first
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    if last < total:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()
    if part.first > 0 and not config.tie_word_embeddings:
        model.model.embed_tokens = nn.Identity()
    return model


def part_names(config: PretrainedConfig, manifest: Manifest, part: Part) -> list[str]:
    """Names of every tensor in a shard that ``part`` reads, as stored in the shard."""
    model = part_skeleton(config, part)
    rename = functools.partial(global_name, first=part.first)
    quantized = {rename(n) for n in quantized_weights(model)}
    names = [
        f"{spec.name}/{piece}"
        for spec in manifest.quantized()
        if spec.name in quantized
        for piece in ("codes", "minimum", "step")
    ]
    _, full = split_parameters(model)
    return names + [rename(n) for n in full]


def global_name(name: str, first: int) -> str:
    """The full model's name of a part's tensor ``name``, for a part starting at ``first``."""
    match = _LAYER.match(name)
    if match is None or not first:
        return name
    return f"{match[1]}{int(match[2]) + first}{match[3]}"


def set_exchange(model: nn.Module, average: Average | None) -> None:
    """Fuse ``model``'s layers through ``average``, or run alone with ``None``."""
    for module in model.modules():
        if isinstance(module, PackedWeight):
            module.exchange = average


def packed_bytes(model: nn.Module) -> int:
    """Device memory held by the packed weights."""
    weights = {id(m): m for m in model.modules() if isinstance(m, PackedWeight)}
    return sum(m.nbytes for m in weights.values())


def _swap_layers(
    model: nn.Module, packed: Mapping[str, PackedWeight], rename: Callable[[str], str]
) -> None:
    """Replace every quantized linear and embedding with its packed version (shared if tied)."""
    names = {id(w): rename(name) for name, w in quantized_weights(model).items()}
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, (nn.Linear, nn.Embedding)) or id(child.weight) not in names:
                continue
            weight = packed[names[id(child.weight)]]
            if isinstance(child, nn.Embedding):
                replacement = PackedEmbedding(weight, child.padding_idx)
            else:
                replacement = PackedLinear(weight, child.bias is not None)
            setattr(parent, child_name, replacement)


def _set_parameter(model: nn.Module, name: str, value: Tensor) -> None:
    module_name, _, attribute = name.rpartition(".")
    module = model.get_submodule(module_name)
    module._parameters[attribute] = nn.Parameter(value, requires_grad=False)


def _reset_rotary(model: nn.Module, device: str) -> None:
    """Rebuild rotary embeddings, whose buffers are computed, not stored."""
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if "Rotary" in type(child).__name__:
                with torch.device(device):
                    setattr(parent, child_name, type(child)(config=child.config))
