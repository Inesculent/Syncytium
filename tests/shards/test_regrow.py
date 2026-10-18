import math
import random
from itertools import combinations

import pytest
import torch

from syncytium.models import split_parameters
from syncytium.quant import (
    Scheme,
    ambiguous,
    child_codes,
    fit_grid,
    parent_codes,
    parent_from_residues,
    quantize_children,
)
from syncytium.quant.grid import Grid
from syncytium.shards import (
    MANIFEST_NAME,
    Redundancy,
    RegrowthManifest,
    ShardError,
    rebuild_from_files,
    regrow_codes,
    regrow_shards,
    regrowth_data,
    shard_name,
    write_children,
    write_regrowth,
)
from syncytium.shards.cli import regrow_main
from syncytium.shards.manifest import file_sha256
from syncytium.shards.packing import packed_size
from syncytium.shards.regrow import (
    decode_residues,
    regrowth_manifest_name,
    regrowth_name,
    residue_shares,
)

BITS = [2, 3, 4]
PHASES = [0.0, 0.5]
NEIGHBOUR, RESIDUE = Redundancy.NEIGHBOUR, Redundancy.RESIDUE


def children_of(groups, bits, num_children, phase):
    return quantize_children(groups, fit_grid(groups, bits), num_children, phase)


def survivors(children, lost):
    return {k: codes for k, codes in enumerate(children) if k not in lost}


def loss_patterns(num_children, tolerated, sample=None, seed=0):
    patterns = list(combinations(range(num_children), tolerated))
    if sample is None or len(patterns) <= sample:
        return patterns
    return random.Random(seed).sample(patterns, sample)


# H14: every loss pattern each scheme is designed for, at N = 4, 8 and 16.


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("num_children", [4, 8, 16])
@pytest.mark.parametrize("phase", PHASES)
def test_neighbour_bits_regrow_every_single_loss(groups, bits, num_children, phase):
    children = children_of(groups, bits, num_children, phase)
    assert all(c.min() == 0 and c.max() == (1 << bits) - 1 for c in children)  # both ends
    held = regrowth_data(children, NEIGHBOUR)
    for index in range(num_children):
        available = survivors(children, {index})
        kept = {k: data for k, data in enumerate(held) if k != index}
        regrown = regrow_codes(available, kept, [index], NEIGHBOUR)
        assert regrown.keys() == {index}
        assert torch.equal(regrown[index], children[index])


# Every pattern for N = 4 and 8; a seeded sample of the 12,870 patterns of eight for N = 16.
PATTERNS = [(4, 1, None), (4, 2, None), (4, 3, None), (8, 1, None), (8, 4, None), (16, 1, None)]
PATTERNS += [(16, 8, 25)]


@pytest.mark.parametrize(("num_children", "tolerated", "sample"), PATTERNS)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("phase", PHASES)
def test_coded_residue_regrows_every_loss_of_t(
    groups, num_children, tolerated, sample, bits, phase
):
    children = children_of(groups, bits, num_children, phase)
    assert all(c.min() == 0 and c.max() == (1 << bits) - 1 for c in children)  # both ends
    held = regrowth_data(children, RESIDUE, tolerated)
    expected_parent = parent_codes(children)
    for lost in loss_patterns(num_children, tolerated, sample, seed=bits):
        available = survivors(children, set(lost))
        kept = {k: data for k, data in enumerate(held) if k not in lost}
        regrown = regrow_codes(available, kept, lost, RESIDUE, tolerated)
        assert all(torch.equal(regrown[k], children[k]) for k in lost)
        # The same shares rebuild the exact parent from any one survivor.
        survivor, codes = next(iter(available.items()))
        residue = decode_residues(kept, codes.numel(), num_children, tolerated)
        parent = parent_from_residues(residue.reshape(codes.shape), survivor, codes, num_children)
        assert torch.equal(parent, expected_parent)


def test_neighbour_bits_cannot_survive_losing_a_child_and_its_holder(groups):
    children = children_of(groups, 3, 4, 0.5)
    held = regrowth_data(children, NEIGHBOUR)
    for lost in ([1, 2], [3, 0]):
        kept = {k: data for k, data in enumerate(held) if k not in lost}
        with pytest.raises(ShardError, match="both lost"):
            regrow_codes(survivors(children, set(lost)), kept, lost, NEIGHBOUR)


def test_neighbour_bits_regrow_losses_that_are_not_adjacent(groups):
    children = children_of(groups, 3, 8, 0.5)
    held = regrowth_data(children, NEIGHBOUR)
    lost = [0, 2, 5]
    kept = {k: data for k, data in enumerate(held) if k not in lost}
    regrown = regrow_codes(survivors(children, set(lost)), kept, lost, NEIGHBOUR)
    assert all(torch.equal(regrown[k], children[k]) for k in lost)


def test_coded_residue_cannot_survive_more_than_t_losses(groups):
    children = children_of(groups, 3, 8, 0.5)
    held = regrowth_data(children, RESIDUE, 2)
    lost = [1, 4, 6]
    kept = {k: data for k, data in enumerate(held) if k not in lost}
    with pytest.raises(ShardError, match="at most 2"):
        regrow_codes(survivors(children, set(lost)), kept, lost, RESIDUE, 2)


@pytest.mark.parametrize(
    ("kind", "num_children", "tolerated"),
    [(NEIGHBOUR, 4, 2), (RESIDUE, 4, 0), (RESIDUE, 4, 4), (NEIGHBOUR, 1, 1)],
)
def test_impossible_redundancy_is_rejected(groups, kind, num_children, tolerated):
    children = children_of(groups, 3, num_children, 0.0)
    with pytest.raises(ShardError):
        regrowth_data(children, kind, tolerated)


# H14: storage per weight per child.


@pytest.mark.parametrize("num_children", [4, 8, 16])
@pytest.mark.parametrize("phase", PHASES)
def test_neighbour_bits_cost_two_in_n_bits_per_weight(num_children, phase):
    # One weight per parent code, so the residues are exactly uniform.
    bits = 3
    fine = torch.arange(num_children * ((1 << bits) - 1), dtype=torch.float64)
    x = (fine + 0.5 - phase) / num_children
    one = torch.ones(1, dtype=torch.float64)
    grid = Grid(0 * one, one, bits)
    children = [child_codes(x, grid, k, num_children, phase) for k in range(num_children)]
    for index, packed in enumerate(regrowth_data(children, NEIGHBOUR)):
        lost = (index - 1) % num_children  # child i holds the bits of child i - 1
        count = int(ambiguous(survivors(children, {lost}), lost, num_children).sum())
        assert count * num_children == 2 * x.numel()
        assert packed.numel() == packed_size(count, 1)


@pytest.mark.parametrize("num_children", [4, 8, 16])
def test_neighbour_bits_cost_about_two_in_n_on_weights(groups, num_children):
    # Each group's minimum and maximum have residue 0, where children 0 and N - 1 are
    # ambiguous, so their bits cost about 2/g more; the others cost about 2/N.
    children = children_of(groups, 3, num_children, 0.5)
    costs = [p.numel() * 8 / groups.numel() for p in regrowth_data(children, NEIGHBOUR)]
    by_child = costs[1:] + costs[:1]  # child i holds the bits of child i - 1
    extra = 2 / groups.shape[1]
    for cost in (by_child[0], by_child[-1]):
        assert 2 / num_children < cost < 2 / num_children + extra + 0.01
    assert all(cost == pytest.approx(2 / num_children, abs=0.025) for cost in by_child[1:-1])
    assert sum(costs) / num_children == pytest.approx(2 / num_children, rel=0.05)


# Pre-registered table of bits per weight per child (H14), to two decimals.
COSTS = [(4, 1, 0.67), (4, 2, 1.00), (8, 1, 0.43), (8, 4, 0.75), (16, 1, 0.27), (16, 8, 0.50)]


@pytest.mark.parametrize(("num_children", "tolerated", "table"), COSTS)
def test_coded_residue_costs_log_n_over_survivors_bits_per_weight(
    groups, num_children, tolerated, table
):
    children = children_of(groups, 3, num_children, 0.5)
    shares = residue_shares(children, tolerated)
    per_weight = shares.shape[1] * 8 / groups.numel()
    assert per_weight == pytest.approx(math.log2(num_children) / (num_children - tolerated), 1e-3)
    assert per_weight == pytest.approx(table, abs=0.005)


# Shard files.


def write(model, directory, scheme):
    return write_children(*split_parameters(model), scheme, "tiny", directory)


def hashes(directory, names):
    return {name: file_sha256(directory / name) for name in names}


def lose(directory, kind, lost):
    for index in lost:
        (directory / shard_name(index)).unlink()
        (directory / regrowth_name(kind, index)).unlink()


def rebuild_all(directory, manifest):
    shards = {k: directory / shard_name(k) for k in range(manifest.scheme.num_children)}
    return rebuild_from_files(shards, manifest)  # checks the parent hash


SCHEMES = [
    Scheme(bits=3, group_size=32, num_children=4),
    Scheme(bits=2, group_size=32, num_children=8, phase=0.5, rotation_seed=7),
    Scheme(bits=4, group_size=32, num_children=16, phase=0.5),
]


@pytest.fixture(params=SCHEMES, ids=lambda s: f"{s.bits}bit-N{s.num_children}-phase{s.phase}")
def shard_set(request, tiny_model, tmp_path):
    manifest = write(tiny_model, tmp_path, request.param)
    return tmp_path, manifest


def test_regrowth_data_sits_beside_unchanged_shards(shard_set):
    directory, manifest = shard_set
    n = manifest.scheme.num_children
    before = hashes(directory, [MANIFEST_NAME, *(shard_name(k) for k in range(n))])
    for kind, tolerated in ((NEIGHBOUR, 1), (RESIDUE, n // 2)):
        sidecar = write_regrowth(directory, kind, tolerated)
        assert RegrowthManifest.load(directory / regrowth_manifest_name(kind)) == sidecar
        assert sidecar.parent_sha256 == manifest.parent_sha256
        assert sidecar.file_sha256 == tuple(
            file_sha256(directory / regrowth_name(kind, k)) for k in range(n)
        )
    assert hashes(directory, before) == before
    rebuild_all(directory, manifest)


def test_every_single_lost_shard_regrows_from_neighbour_bits(shard_set):
    directory, manifest = shard_set
    write_regrowth(directory, NEIGHBOUR)
    for index in range(manifest.scheme.num_children):
        lose(directory, NEIGHBOUR, [index])
        assert regrow_shards(directory, [index], NEIGHBOUR) == [directory / shard_name(index)]
        assert file_sha256(directory / shard_name(index)) == manifest.shard_sha256[index]
    rebuild_all(directory, manifest)


def test_t_lost_shards_regrow_from_the_coded_residue(shard_set):
    directory, manifest = shard_set
    n = manifest.scheme.num_children
    sidecar = write_regrowth(directory, RESIDUE, n // 2)
    for lost in (range(n // 2), range(n // 2, n), random.Random(n).sample(range(n), n // 2)):
        lose(directory, RESIDUE, lost)
        regrow_shards(directory, lost, RESIDUE)
        for index in lost:
            assert file_sha256(directory / shard_name(index)) == manifest.shard_sha256[index]
            path = directory / regrowth_name(RESIDUE, index)
            assert file_sha256(path) == sidecar.file_sha256[index]
    rebuild_all(directory, manifest)


@pytest.fixture
def four(tiny_model, tmp_path):
    manifest = write(tiny_model, tmp_path, SCHEMES[0])
    write_regrowth(tmp_path, NEIGHBOUR)
    write_regrowth(tmp_path, RESIDUE, 2)
    return tmp_path, manifest


def test_no_drift_after_ten_rounds_in_the_shard_format(four):
    # H15: the parent hash is checked by every regrowth and by the final rebuild.
    directory, manifest = four
    rng = random.Random(0)
    for _ in range(10):
        for index in range(4):
            lose(directory, NEIGHBOUR, [index])
            regrow_shards(directory, [index], NEIGHBOUR)
        lost = rng.sample(range(4), 2)
        lose(directory, RESIDUE, lost)
        regrow_shards(directory, lost, RESIDUE)
    assert all(file_sha256(directory / shard_name(k)) == manifest.shard_sha256[k] for k in range(4))
    rebuild_all(directory, manifest)


def test_adjacent_shard_losses_fail_loudly(four):
    directory, _ = four
    lose(directory, NEIGHBOUR, [3, 0])
    with pytest.raises(ShardError, match="children 3 and 0 are both lost"):
        regrow_shards(directory, [3, 0], NEIGHBOUR)
    assert not (directory / shard_name(0)).exists()  # nothing half-written
    regrow_shards(directory, [3, 0], RESIDUE)
    rebuild_all(directory, four[1])


def test_too_many_shard_losses_fail_loudly(four):
    directory, _ = four
    with pytest.raises(ShardError, match="at most 2"):
        regrow_shards(directory, [0, 1, 2], RESIDUE)


def test_a_corrupt_lost_shard_is_replaced_without_being_read(four):
    directory, manifest = four
    (directory / shard_name(2)).write_bytes(b"not a shard")
    regrow_shards(directory, [2], NEIGHBOUR)
    assert file_sha256(directory / shard_name(2)) == manifest.shard_sha256[2]


def test_a_tampered_survivor_regrowth_file_is_rejected(four):
    directory, _ = four
    path = directory / regrowth_name(NEIGHBOUR, 2)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    lose(directory, NEIGHBOUR, [1])
    with pytest.raises(ShardError, match="regrowth sidecar"):
        regrow_shards(directory, [1], NEIGHBOUR)


def test_a_sidecar_from_another_shard_set_is_rejected(four):
    directory, _ = four
    path = directory / regrowth_manifest_name(RESIDUE)
    sidecar = RegrowthManifest.load(path)
    RegrowthManifest(**{**sidecar.__dict__, "parent_sha256": "0" * 64}).save(path)
    with pytest.raises(ShardError, match="different shard set"):
        regrow_shards(directory, [1], RESIDUE)


def test_regrow_command_line(tiny_model, tmp_path, capsys):
    manifest = write(tiny_model, tmp_path, SCHEMES[0])
    regrow_main([str(tmp_path), "--kind", "residue", "--tolerate", "2"])
    assert "regrow-residue-3.safetensors" in capsys.readouterr().out
    lose(tmp_path, RESIDUE, [0, 2])
    regrow_main([str(tmp_path), "--kind", "residue", "--lost", "0", "2"])
    assert "regrew child-2.safetensors" in capsys.readouterr().out
    rebuild_all(tmp_path, manifest)
