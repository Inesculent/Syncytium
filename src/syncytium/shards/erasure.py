"""Reed-Solomon erasure coding over GF(2^8), for the coded residue (H14).

``k`` blocks of data bytes become ``n`` shares (n ≤ 255) such that any ``k`` shares recover
the data. The generator is systematic: the first ``k`` shares are the data blocks themselves,
and share k + i is Σⱼ Cᵢⱼ dⱼ for the Cauchy matrix Cᵢⱼ = 1/(xᵢ + yⱼ) with xᵢ = k + i and
yⱼ = j. Every square submatrix of a Cauchy matrix is invertible, so every k rows of the
generator are, which is the MDS property.

Field arithmetic uses log and antilog tables of GF(2^8) with the polynomial
x^8 + x^4 + x^3 + x^2 + 1. Addition is XOR.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

ORDER = 255  # multiplicative group of GF(2^8)
POLYNOMIAL = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1; 2 generates the multiplicative group


def _tables() -> tuple[list[int], list[int]]:
    """Antilog table (doubled, so products need no reduction mod 255) and log table."""
    antilog, log = [0] * (2 * ORDER), [0] * (ORDER + 1)
    value = 1
    for power in range(ORDER):
        antilog[power] = antilog[power + ORDER] = value
        log[value] = power
        value <<= 1
        if value > ORDER:
            value ^= POLYNOMIAL
    return antilog, log


ANTILOG, LOG = _tables()


def multiply(a: int, b: int) -> int:
    """Product of two field elements."""
    if a == 0 or b == 0:
        return 0
    return ANTILOG[LOG[a] + LOG[b]]


def inverse(a: int) -> int:
    """Multiplicative inverse of a nonzero field element.

    Raises:
        ZeroDivisionError: If ``a`` is zero.
    """
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(2^8)")
    return ANTILOG[ORDER - LOG[a]]


def _product_table() -> Tensor:
    """All 256 x 256 products, so scaling a byte tensor is one lookup."""
    log = torch.tensor(LOG)
    antilog = torch.tensor(ANTILOG, dtype=torch.uint8)
    table = antilog[log.unsqueeze(1) + log.unsqueeze(0)]
    table[0, :] = 0
    table[:, 0] = 0
    return table


PRODUCTS = _product_table()


def generator(num_shares: int, num_data: int) -> list[list[int]]:
    """The systematic Cauchy generator: ``num_shares`` rows of ``num_data`` coefficients.

    Raises:
        ValueError: Unless 1 ≤ ``num_data`` ≤ ``num_shares`` ≤ 255.
    """
    if not 1 <= num_data <= num_shares <= ORDER:
        raise ValueError(f"need 1 ≤ k ≤ n ≤ {ORDER}, got k = {num_data}, n = {num_shares}")
    identity = [[int(row == col) for col in range(num_data)] for row in range(num_data)]
    # x_i = k + i and y_j = j are distinct, so x_i + y_j (XOR) is never zero.
    cauchy = [
        [inverse((num_data + row) ^ col) for col in range(num_data)]
        for row in range(num_shares - num_data)
    ]
    return identity + cauchy


def encode(data: Tensor, num_shares: int) -> Tensor:
    """Encode ``k`` data blocks into ``num_shares`` shares.

    Args:
        data: ``uint8`` tensor of shape ``(k, length)``.
        num_shares: Number of shares n.

    Returns:
        ``uint8`` tensor of shape ``(n, length)``; its first k rows equal ``data``.
    """
    return _combine(generator(num_shares, data.shape[0]), data)


def decode(shares: Mapping[int, Tensor], num_data: int, num_shares: int) -> Tensor:
    """Recover the ``k`` data blocks from any ``k`` of the ``n`` shares.

    Args:
        shares: Available shares (1-D ``uint8``, equal lengths), keyed by share index.
        num_data: Number of data blocks k.
        num_shares: Number of shares n.

    Returns:
        ``uint8`` tensor of shape ``(k, length)``.

    Raises:
        ValueError: If fewer than k shares are given or an index is out of range.
    """
    if len(shares) < num_data:
        raise ValueError(f"need {num_data} of the {num_shares} shares, got {len(shares)}")
    if not all(0 <= index < num_shares for index in shares):
        raise ValueError(f"share indices must be in [0, {num_shares}), got {sorted(shares)}")
    chosen = sorted(shares)[:num_data]
    rows = generator(num_shares, num_data)
    solve = _invert([rows[index] for index in chosen])
    return _combine(solve, torch.stack([shares[index] for index in chosen]))


def _combine(matrix: list[list[int]], blocks: Tensor) -> Tensor:
    """Matrix product over GF(2^8): row i of the result is Σⱼ matrix[i][j] · blocks[j]."""
    blocks = blocks.to(torch.uint8)
    indices = blocks.long()
    out = torch.zeros(len(matrix), blocks.shape[1], dtype=torch.uint8)
    for row, coefficients in enumerate(matrix):
        for block, coefficient in zip(indices, coefficients, strict=True):
            if coefficient:
                out[row] ^= PRODUCTS[coefficient][block]
    return out


def _invert(matrix: list[list[int]]) -> list[list[int]]:
    """Inverse of a square matrix over GF(2^8), by Gauss-Jordan elimination.

    Raises:
        ValueError: If the matrix is singular (never for k rows of :func:`generator`).
    """
    size = len(matrix)
    rows = [row[:] + [int(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = next((r for r in range(col, size) if rows[r][col]), None)
        if pivot is None:
            raise ValueError("matrix is singular over GF(2^8)")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rows[col] = _scale_row(rows[col], inverse(rows[col][col]))
        for r in range(size):
            if r != col and rows[r][col]:
                rows[r] = _add_rows(rows[r], _scale_row(rows[col], rows[r][col]))
    return [row[size:] for row in rows]


def _scale_row(row: list[int], factor: int) -> list[int]:
    return [multiply(value, factor) for value in row]


def _add_rows(a: list[int], b: list[int]) -> list[int]:
    return [x ^ y for x, y in zip(a, b, strict=True)]
