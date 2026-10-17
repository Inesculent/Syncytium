import torch

from holograph.quant import Scheme, simulate_child, simulate_rebuild
from holograph.sweep import Condition, Kind, plan
from holograph.sweep.config import GridConfig


def grid(**overrides):
    settings = {"bits": (4,), "group_sizes": (64,), "children": (4,)} | overrides
    return GridConfig(**settings)


def child(n, k, phase=0.0, bits=4):
    return Condition(Kind.CHILD, Scheme(bits, 64, n, phase), (k,))


def test_plan_starts_with_the_bf16_baseline():
    assert plan(grid())[0] == Condition(Kind.BF16)


def test_plan_covers_children_rebuild_and_partials():
    kinds = [c.kind for c in plan(grid(max_missing=1))[1:]]
    assert kinds.count(Kind.CHILD) == 4
    assert kinds.count(Kind.REBUILD) == 1
    assert kinds.count(Kind.PARTIAL) == 4


def test_max_missing_never_removes_every_child():
    partials = [c for c in plan(grid(children=(2,), max_missing=5)) if c.kind == Kind.PARTIAL]
    assert all(len(c.indices) >= 1 for c in partials)


def test_condition_ids_are_unique():
    conditions = plan(grid(bits=(2, 4), children=(2, 4, 8), phases=(0.0, 0.5), rtn_bits=(4,)))
    assert len({c.id for c in conditions}) == len(conditions)


def test_nested_children_share_a_key():
    assert child(4, 1).key == child(8, 2).key
    assert child(4, 1).key != child(8, 1).key


def test_centered_children_coincide_with_finer_uncentered_ones():
    assert child(4, 0, phase=0.5).key == child(8, 1).key


def test_round_to_nearest_matches_the_middle_child():
    rtn = Condition(Kind.RTN, Scheme(4, 64, 1, phase=0.5), (0,))
    assert rtn.key == child(4, 2).key


def test_partial_rebuild_of_one_child_matches_that_child():
    partial = Condition(Kind.PARTIAL, Scheme(4, 64, 2), (1,))
    assert partial.key == child(2, 1).key


def test_bit_width_is_part_of_the_key():
    assert child(4, 1, bits=3).key != child(4, 1, bits=4).key


def test_transforms_apply_the_right_simulation(groups):
    weight = groups.reshape(64, 512)
    scheme = Scheme(3, 64, 4)
    assert Condition(Kind.BF16).transform() is None
    assert torch.equal(child(4, 1, bits=3).transform()(weight), simulate_child(weight, scheme, 1))
    partial = Condition(Kind.PARTIAL, scheme, (0, 2, 3)).transform()(weight)
    assert torch.equal(partial, simulate_rebuild(weight, scheme, (0, 2, 3)))
