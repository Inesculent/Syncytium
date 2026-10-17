"""Packing b-bit codes densely into bytes.

Codes are laid out back to back, least significant bit first, so a tensor of n codes takes
⌈n b / 8⌉ bytes for any b from 1 to 8 (including 3 bits, which does not align with bytes).
"""

import torch
from torch import Tensor

from syncytium.quant.grid import MAX_BITS

BYTE = 8


def pack(codes: Tensor, bits: int) -> Tensor:
    """Pack ``uint8`` codes below 2^``bits`` into a 1-D ``uint8`` tensor.

    Raises:
        ValueError: If ``bits`` is outside ``[1, 8]`` or a code does not fit in ``bits``.
    """
    _check_bits(bits)
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() and int(flat.max()) >> bits:
        raise ValueError(f"codes do not fit in {bits} bits")
    shifts = torch.arange(bits, dtype=torch.uint8)
    stream = ((flat.unsqueeze(-1) >> shifts) & 1).reshape(-1)
    stream = torch.nn.functional.pad(stream, (0, -stream.numel() % BYTE))
    return _merge_bits(stream.reshape(-1, BYTE))


def unpack(packed: Tensor, bits: int, count: int) -> Tensor:
    """Recover ``count`` codes from the output of :func:`pack`.

    Raises:
        ValueError: If ``packed`` holds fewer than ``count`` codes.
    """
    _check_bits(bits)
    if packed.numel() * BYTE < count * bits:
        raise ValueError(f"{packed.numel()} bytes cannot hold {count} codes of {bits} bits")
    shifts = torch.arange(BYTE, dtype=torch.uint8)
    stream = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)[: count * bits]
    return _merge_bits(stream.reshape(count, bits))


def packed_size(count: int, bits: int) -> int:
    """Bytes needed to pack ``count`` codes of ``bits`` bits."""
    return -(-count * bits // BYTE)


def _merge_bits(bit_columns: Tensor) -> Tensor:
    """Combine rows of bits (least significant first) into one ``uint8`` per row."""
    merged = torch.zeros(bit_columns.shape[0], dtype=torch.uint8)
    for position in range(bit_columns.shape[1]):
        merged |= bit_columns[:, position] << position
    return merged


def _check_bits(bits: int) -> None:
    if not 1 <= bits <= MAX_BITS:
        raise ValueError(f"bits must be in [1, {MAX_BITS}], got {bits}")
