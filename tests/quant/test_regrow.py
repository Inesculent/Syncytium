import hashlib

import pytest
import torch

from syncytium.quant import (
    Grid,
    ambiguous,
    approximate_regrowth,
    check_staggered,
    child_codes,
    child_weights,
    fit_grid,
    neighbour_bits,
    parent_codes,
    parent_from_residues,
    parent_weights,
    quantize_children,
    regrow_from_residues,
    regrow_with_neighbour_bits,
    residues,
)

BITS = [2, 3, 4]
CHILDREN = [4, 8, 16]
PHASES = [0.0, 0.5]


def unit_grid(bits):
    one = torch.ones(1, dtype=torch.float64)
    return Grid(minimum=0 * one, step=one, bits=bits)


def every_fine_code(bits, num_children, phase, ends=True):
    """One weight per parent code Q in [0, N(2^b - 1)), at the centre of its fine step, so the
    residues are exactly uniform; with ``ends``, also both ends of the range exactly."""
    fine = torch.arange(num_children * ((1 << bits) - 1), dtype=torch.float64)
    x = (fine + 0.5 - phase) / num_children
    if ends:
        x = torch.cat([x, torch.tensor([0.0, (1 << bits) - 1.0], dtype=torch.float64)])
    grid = unit_grid(bits)
    return [child_codes(x, grid, k, num_children, phase) for k in range(num_children)]


def without(children, index):
    return {k: codes for k, codes in enumerate(children) if k != index}


def parent_hash(children):
    codes = parent_codes(children).numpy().astype("<i4")
    return hashlib.sha256(codes.tobytes()).hexdigest()


@pytest.mark.parametrize("num_children", [2, 3, *CHILDREN])
@pytest.mark.parametrize("phase", PHASES)
def test_lost_code_is_ambiguous_at_exactly_two_in_n_weights(num_children, phase):
    children = every_fine_code(3, num_children, phase, ends=False)
    for index in range(num_children):
        mask = ambiguous(without(children, index), index, num_children)
        assert mask.sum() * num_children == 2 * mask.numel()


def staggered_at(children):
    """Per weight: whether the codes are q_k = ⌊(Q + k)/N⌋ for some Q (range ignored)."""
    codes = torch.stack([c.to(torch.int16) for c in children])
    steps = codes.diff(dim=0)
    return ((steps == 0) | (steps == 1)).all(dim=0) & (codes[-1] - codes[0] <= 1)


@pytest.mark.parametrize("num_children", [2, 3, *CHILDREN])
@pytest.mark.parametrize("phase", PHASES)
def test_lost_code_has_two_adjacent_candidates_exactly_where_flagged(num_children, phase):
    # Exhaustive over every parent code: replacing the lost code by its neighbours on the
    # integer line keeps the set staggered exactly at the weights survivors flag. At the
    # range ends one of the two candidates can be -1 or 2^b; a bit is stored there too.
    children = every_fine_code(3, num_children, phase)
    for index in range(num_children):
        mask = ambiguous(without(children, index), index, num_children)
        valid = []
        for shift in (-1, 1):
            moved = list(children)
            moved[index] = children[index].to(torch.int16) + shift
            valid.append(staggered_at(moved))
        assert torch.equal(valid[0] | valid[1], mask)
        assert not torch.any(valid[0] & valid[1])


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", CHILDREN)
@pytest.mark.parametrize("phase", PHASES)
def test_neighbour_bits_regrow_every_single_loss(groups, bits, num_children, phase):
    children = quantize_children(groups, fit_grid(groups, bits), num_children, phase)
    assert all(c.min() == 0 and c.max() == (1 << bits) - 1 for c in children)  # both ends
    for index in range(num_children):
        stored = neighbour_bits(children, index)
        regrown = regrow_with_neighbour_bits(without(children, index), index, stored, num_children)
        assert torch.equal(regrown, children[index])


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", [2, 3, *CHILDREN])
@pytest.mark.parametrize("phase", PHASES)
def test_neighbour_bits_regrow_every_fine_code(bits, num_children, phase):
    children = every_fine_code(bits, num_children, phase)
    for index in range(num_children):
        stored = neighbour_bits(children, index)
        regrown = regrow_with_neighbour_bits(without(children, index), index, stored, num_children)
        assert torch.equal(regrown, children[index])


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", CHILDREN)
@pytest.mark.parametrize("phase", PHASES)
def test_residues_and_any_survivor_give_every_child_and_the_parent(
    groups, bits, num_children, phase
):
    children = quantize_children(groups, fit_grid(groups, bits), num_children, phase)
    residue = residues(children)
    for survivor, codes in enumerate(children):
        parent = parent_from_residues(residue, survivor, codes, num_children)
        assert torch.equal(parent, parent_codes(children))
        for index, expected in enumerate(children):
            regrown = regrow_from_residues(residue, survivor, codes, index, num_children)
            assert torch.equal(regrown, expected)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("phase", PHASES)
def test_residues_regrow_every_fine_code(bits, phase):
    children = every_fine_code(bits, 8, phase)
    residue = residues(children)
    for index, expected in enumerate(children):
        survivor = (index + 3) % 8
        regrown = regrow_from_residues(residue, survivor, children[survivor], index, 8)
        assert torch.equal(regrown, expected)


@pytest.mark.parametrize("num_children", CHILDREN)
def test_no_drift_after_ten_rounds_of_regrowing_every_member(groups, num_children):
    children = quantize_children(groups, fit_grid(groups, 3), num_children, 0.5)
    expected = parent_hash(children)
    for _ in range(10):
        for index in range(num_children):
            stored = neighbour_bits(children, index)
            children[index] = regrow_with_neighbour_bits(
                without(children, index), index, stored, num_children
            )
    assert parent_hash(children) == expected


def test_regrowth_needs_both_neighbours(groups):
    children = quantize_children(groups, fit_grid(groups, 3), 4)
    stored = neighbour_bits(children, 1)
    available = {0: children[0], 3: children[3]}
    with pytest.raises(ValueError, match="neighbours 0 and 2"):
        regrow_with_neighbour_bits(available, 1, stored, 4)


def test_wrong_number_of_bits_is_rejected(groups):
    children = quantize_children(groups, fit_grid(groups, 3), 4)
    stored = neighbour_bits(children, 2)
    with pytest.raises(ValueError, match="ambiguous codes"):
        regrow_with_neighbour_bits(without(children, 2), 2, stored[:-1], 4)


def test_unstaggered_children_are_rejected(groups):
    children = quantize_children(groups, fit_grid(groups, 3), 4)
    swapped = [children[0], children[3], children[2], children[1]]
    with pytest.raises(ValueError, match="not staggered"):
        check_staggered(swapped)
    with pytest.raises(ValueError, match="not staggered"):
        residues(swapped)
    with pytest.raises(ValueError, match="not staggered"):
        neighbour_bits(swapped, 1)
    with pytest.raises(ValueError, match="at least two"):
        neighbour_bits(children[:1], 0)


# Negative control ("Known when fixing"): N, then ambiguous, wrong, regrown-child MSE ratio and
# rebuild MSE ratio as reported in the pre-registration.
KNOWN = [(4, 0.50, 0.25, 1.75, 4.0), (8, 0.25, 0.125, 1.19, 2.5)]


def monte_carlo(num_children, phase, seed=0, count=100_000, bits=3):
    """Uniform weights away from the ends of the range, with one generator for the coins."""
    generator = torch.Generator().manual_seed(seed)
    x = 1 + ((1 << bits) - 3) * torch.rand(count, generator=generator, dtype=torch.float64)
    grid = unit_grid(bits)
    children = [child_codes(x, grid, k, num_children, phase) for k in range(num_children)]
    return x, grid, children, generator


def mse(estimate, x):
    return float((estimate - x).pow(2).mean())


@pytest.mark.parametrize(("num_children", "unsure", "wrong", "child", "rebuild"), KNOWN)
@pytest.mark.parametrize("phase", PHASES)
def test_approximate_regrowth_loses_precision(num_children, unsure, wrong, child, rebuild, phase):
    x, grid, children, generator = monte_carlo(num_children, phase)
    index = 1
    available = without(children, index)
    guessed = approximate_regrowth(available, index, num_children, grid.bits, generator)
    assert float(ambiguous(available, index, num_children).double().mean()) == pytest.approx(
        unsure, abs=0.01
    )
    assert float((guessed != children[index]).double().mean()) == pytest.approx(wrong, abs=0.01)
    original = mse(child_weights(children[index], grid, index, num_children, phase), x)
    regrown = mse(child_weights(guessed, grid, index, num_children, phase), x)
    assert regrown / original == pytest.approx(child, rel=0.05)
    exact = mse(parent_weights(parent_codes(children), grid, num_children, phase), x)
    approximate = parent_codes([*available.values(), guessed])
    ratio = mse(parent_weights(approximate, grid, num_children, phase), x) / exact
    assert ratio == pytest.approx(rebuild, rel=0.05)


@pytest.mark.parametrize("num_children", [4, 8])
def test_repeated_approximate_regrowth_breaks_the_fleet(num_children):
    # The pre-registration measured about 116x (N = 4) and 118x (N = 8) after five rounds.
    x, grid, children, generator = monte_carlo(num_children, 0.5)
    exact = mse(parent_weights(parent_codes(children), grid, num_children, 0.5), x)
    for _ in range(5):
        for index in range(num_children):
            children[index] = approximate_regrowth(
                without(children, index), index, num_children, grid.bits, generator
            )
    drifted = mse(parent_weights(parent_codes(children), grid, num_children, 0.5), x)
    assert 80 < drifted / exact < 160
