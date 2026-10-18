import pytest
import torch

from syncytium.quant import Estimator, Scheme, simulate_child, simulate_rebuild
from syncytium.tasks import Kind, TaskCondition, plan

SCHEME = Scheme(bits=3, group_size=64, num_children=4, phase=0.5)


def test_plan_covers_every_child_group_and_drop_once():
    conditions = plan(4, drop=True)
    ids = [c.id for c in conditions]
    assert len(ids) == len(set(ids)) == 1 + 4 + (1 + 4 + 6) + 4
    assert ids[:6] == ["bf16", "child/k0", "child/k1", "child/k2", "child/k3", "group/k0,1,2,3"]
    assert ids[-4:] == ["drop/k0", "drop/k1", "drop/k2", "drop/k3"]


def test_plan_without_drops():
    assert not [c for c in plan(4, drop=False) if c.kind == Kind.DROP]


def test_a_drop_leaves_the_other_members():
    drop = TaskCondition(Kind.DROP, (0, 1, 2, 3), dropped=2)
    assert drop.survivors == TaskCondition(Kind.GROUP, (0, 1, 3))
    with pytest.raises(ValueError, match="two weight states"):
        drop.transform(SCHEME, Estimator.MEAN)
    with pytest.raises(ValueError, match="drops no child"):
        _ = TaskCondition(Kind.GROUP, (0, 1)).survivors


def test_transforms_match_the_simulated_weights(groups):
    weight = groups.reshape(16, -1)
    child = TaskCondition(Kind.CHILD, (2,)).transform(SCHEME, Estimator.MEAN)
    assert torch.equal(child("w", weight), simulate_child(weight, SCHEME, 2))
    trio = TaskCondition(Kind.GROUP, (0, 1, 3)).transform(SCHEME, Estimator.MEAN)
    expected = simulate_rebuild(weight, SCHEME, (0, 1, 3), Estimator.MEAN)
    assert torch.equal(trio("w", weight), expected)
    assert TaskCondition(Kind.BF16).transform(SCHEME, Estimator.MEAN) is None


def test_scaled_scheme_needs_scales():
    scaled = Scheme(bits=3, group_size=64, num_children=4, scaling=True)
    with pytest.raises(ValueError, match="no calibration"):
        TaskCondition(Kind.CHILD, (0,)).transform(scaled, Estimator.MEAN)
