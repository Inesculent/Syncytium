"""Regrowing a lost child exactly from the survivors (``docs/preregistration-swarm.md``, H14).

Every child is a fixed function of the parent code Q (``docs/method.md``, "Rebuild" and
"Offset phase"): q_k = ⌊(Q + k)/N⌋. Equivalently, with m = ⌊Q/N⌋ and residue r = Q mod N,
q_k = m + [r + k ≥ N], so the codes of one weight never decrease with k and the last child is
at most one above the first. Two kinds of stored data turn that structure into exact regrowth:

- *Neighbour bits.* Given children k - 1 and k + 1, child k's code is one of two adjacent
  values at exactly the weights where those neighbours differ (2/N of them when residues are
  uniform). One bit per such weight resolves it.
- *Residues.* r together with any one surviving child gives m, hence every child and Q.

:func:`approximate_regrowth` guesses instead of storing anything. It is the negative control
of the pre-registration ("Known when fixing") and is not exact.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from syncytium.quant.rebuild import parent_codes

# Wide enough for a code minus one, and for N up to 255 children summed into a residue.
_WORK = torch.int16


def check_staggered(children: Sequence[Tensor]) -> None:
    """Check that ``children`` are the codes q_k = ⌊(Q + k)/N⌋ of one parent.

    Regrowth relies on this structure; codes that break it (from a different scheme, or
    corrupted) would otherwise regrow into something plausible but wrong.

    Raises:
        ValueError: If fewer than two children are given, or a code drops from one child to
            the next, rises by more than one, or the last child exceeds the first by more
            than one.
    """
    for index in range(len(children)):
        _checked_bounds(children, index)


def neighbour_bits(children: Sequence[Tensor], index: int) -> Tensor:
    """The bits that let child ``index`` be regrown from its neighbours, in position order.

    One bit per weight where the neighbours leave two candidates: 1 if the code is the
    higher one. These are stored with child ``index + 1`` (mod N), so any single loss leaves
    both the neighbours and the bits.

    Args:
        children: Codes of all N children, ordered by child index.
        index: The child the bits describe.

    Returns:
        A 1-D ``uint8`` tensor of zeros and ones, one per ambiguous weight.

    Raises:
        ValueError: If fewer than two children are given, or child ``index`` is not staggered
            with its neighbours (see :func:`check_staggered`).
    """
    low, high = _checked_bounds(children, index)
    return (children[index].to(_WORK) - low)[high != low].to(torch.uint8)


def ambiguous(available: Mapping[int, Tensor], index: int, num_children: int) -> Tensor:
    """Where child ``index``'s neighbours leave two candidate codes, as a ``bool`` mask.

    Survivors compute this from their own codes, so they know how many neighbour bits to
    read and which weights an approximate regrowth has to guess.

    Raises:
        ValueError: If a neighbour is missing.
    """
    low, high = _bounds_or_explain(available, index, num_children)
    return high != low


def regrow_with_neighbour_bits(
    available: Mapping[int, Tensor], index: int, bits: Tensor, num_children: int
) -> Tensor:
    """Regrow child ``index`` exactly from its neighbours and its stored bits.

    Args:
        available: Codes of the surviving children, keyed by child index; must include
            ``index - 1`` and ``index + 1`` (mod N).
        index: The lost child.
        bits: Output of :func:`neighbour_bits` for this child.
        num_children: Number of children N.

    Raises:
        ValueError: If a neighbour is missing, or ``bits`` has the wrong length or gives a
            negative code (so it belongs to other children).
    """
    low, high = _bounds_or_explain(available, index, num_children)
    ambiguous = high != low
    if bits.numel() != int(ambiguous.sum()):
        raise ValueError(
            f"child {index} has {int(ambiguous.sum())} ambiguous codes but {bits.numel()} bits"
        )
    codes = low.clone()
    codes[ambiguous] += bits.reshape(-1).to(_WORK)
    if codes.numel() and int(codes.min()) < 0:
        raise ValueError(f"the bits do not belong to child {index}'s neighbours")
    return codes.to(torch.uint8)


def residues(children: Sequence[Tensor]) -> Tensor:
    """The residues r = Q mod N of the parent codes, as ``uint8`` (N ≤ 255).

    Raises:
        ValueError: If the children are not staggered or N exceeds 255.
    """
    if len(children) > 255:
        raise ValueError(f"residues fit in a byte only for N ≤ 255, got {len(children)}")
    check_staggered(children)
    return (parent_codes(children) % len(children)).to(torch.uint8)


def regrow_from_residues(
    residue: Tensor, survivor: int, codes: Tensor, index: int, num_children: int
) -> Tensor:
    """Regrow child ``index`` from the residues and any one surviving child.

    q_k = m + [r + k ≥ N], so the survivor's codes give m and m gives every child.

    Args:
        residue: Output of :func:`residues`, same shape as ``codes``.
        survivor: Index i of the surviving child.
        codes: Codes of child i.
        index: The child to regrow.
        num_children: Number of children N.
    """
    floor = _floor_quotient(residue, survivor, codes, num_children)
    return (floor + _carry(residue, index, num_children)).to(torch.uint8)


def parent_from_residues(
    residue: Tensor, survivor: int, codes: Tensor, num_children: int
) -> Tensor:
    """The exact parent codes Q = N m + r from the residues and any one surviving child.

    Args:
        residue: Output of :func:`residues`, same shape as ``codes``.
        survivor: Index i of the surviving child.
        codes: Codes of child i.
        num_children: Number of children N.

    Returns:
        ``int32`` codes, equal to :func:`~syncytium.quant.rebuild.parent_codes` of all N.
    """
    floor = _floor_quotient(residue, survivor, codes, num_children).to(torch.int32)
    return num_children * floor + residue.to(torch.int32)


def approximate_regrowth(
    available: Mapping[int, Tensor],
    index: int,
    num_children: int,
    bits: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Negative control: regrow child ``index`` with no stored data, guessing where unsure.

    Each code is one of the two bounds set by its neighbours, picked by a fair coin (the
    bounds coincide where the code is determined), then kept inside [0, 2^b - 1]. This is
    *not* exact: about half the ambiguous codes come out wrong, and repeated rounds drift.

    Args:
        available: Codes of the surviving children, keyed by child index; must include
            ``index - 1`` and ``index + 1`` (mod N).
        index: The lost child.
        num_children: Number of children N.
        bits: Bits per child code, b.
        generator: Source of the coin flips.

    Raises:
        ValueError: If a neighbour is missing.
    """
    low, high = _bounds_or_explain(available, index, num_children)
    coin = torch.rand(low.shape, generator=generator) < 0.5
    return torch.where(coin, high, low).clamp(0, (1 << bits) - 1).to(torch.uint8)


def _bounds(
    available: Sequence[Tensor] | Mapping[int, Tensor], index: int, num_children: int
) -> tuple[Tensor, Tensor]:
    """Lowest and highest code child ``index`` can have, given its two neighbours.

    The bounds come from q_{k-1} ≤ q_k ≤ q_{k+1} and q_{N-1} ≤ q_0 + 1; for staggered
    children they differ by at most one.
    """
    before, after = (index - 1) % num_children, (index + 1) % num_children
    # Across the wrap the neighbour sits on the other side of the step: q_{N-1} ≤ q_0 + 1.
    low = available[before].to(_WORK) - (1 if index == 0 else 0)
    high = available[after].to(_WORK) + (1 if index == num_children - 1 else 0)
    return low, high


def _checked_bounds(children: Sequence[Tensor], index: int) -> tuple[Tensor, Tensor]:
    """Bounds of child ``index``, checked to hold its codes and differ by at most one.

    That is all its exact regrowth needs, so it is cheaper than :func:`check_staggered`.
    """
    if len(children) < 2:
        raise ValueError(f"regrowth needs at least two children, got {len(children)}")
    low, high = _bounds(children, index, len(children))
    code = children[index].to(_WORK)
    if not torch.all((low <= code) & (code <= high) & (high - low <= 1)):
        raise ValueError(f"child {index}'s codes are not staggered with its neighbours")
    return low, high


def _bounds_or_explain(
    available: Mapping[int, Tensor], index: int, num_children: int
) -> tuple[Tensor, Tensor]:
    try:
        return _bounds(available, index, num_children)
    except KeyError as missing:
        raise ValueError(
            f"regrowing child {index} needs its neighbours {(index - 1) % num_children} and "
            f"{(index + 1) % num_children}; child {missing} is lost too"
        ) from None


def _carry(residue: Tensor, index: int, num_children: int) -> Tensor:
    """[r + k ≥ N]: whether child k sits above the step."""
    return (residue.to(_WORK) + index >= num_children).to(_WORK)


def _floor_quotient(residue: Tensor, survivor: int, codes: Tensor, num_children: int) -> Tensor:
    """The quotient m = ⌊Q/N⌋ = q_i - [r + i ≥ N]."""
    return codes.to(_WORK) - _carry(residue, survivor, num_children)
