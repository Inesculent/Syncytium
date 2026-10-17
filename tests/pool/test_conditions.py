from collections import Counter
from dataclasses import replace

import torch

from syncytium.pool import Kind, PoolCondition, plan, random_patterns
from syncytium.quant import Estimator, simulate_child, simulate_rebuild


def test_plan_starts_with_the_references_and_has_unique_ids(grid):
    conditions = plan(grid)
    assert [c.kind for c in conditions[:2]] == [Kind.BF16, Kind.FULL]
    assert len({c.id for c in conditions}) == len(conditions)


def test_every_device_loss_is_planned_for_each_estimator(grid):
    devices = [c for c in plan(grid) if c.kind == Kind.DEVICE]
    stages, children = grid.stages + 1, grid.children
    assert Counter(c.estimator for c in devices) == {
        Estimator.MEAN: stages * children,
        Estimator.MIDPOINT: stages * children,
    }
    assert all(len(c.lost) == 1 for c in devices)


def test_estimator_only_matters_with_a_partly_missing_stage(grid):
    by_kind = {c.kind: c for c in plan(grid)}
    assert not by_kind[Kind.FULL].estimator_matters
    assert not by_kind[Kind.STAGE].estimator_matters  # the stage keeps a single child
    assert by_kind[Kind.DEVICE].estimator_matters  # the stage keeps two of three
    stage = by_kind[Kind.STAGE]
    assert replace(stage, estimator=Estimator.MIDPOINT).key == stage.key


def test_random_patterns_keep_a_child_on_every_stage(grid):
    patterns = random_patterns(grid)
    assert len(set(patterns)) == len(patterns)
    assert all(all(alive) for alive in patterns)
    lost = Counter(sum(grid.children - len(alive) for alive in p) for p in patterns)
    assert lost == {2: grid.samples, 3: grid.samples}


def test_random_patterns_are_reproducible(grid):
    assert random_patterns(grid) == random_patterns(grid)


def test_every_stage_of_a_random_pattern_is_planned_on_its_own(grid):
    conditions = plan(grid)
    keys = {c.key for c in conditions}
    everyone = frozenset(range(grid.children))
    for pattern in (c for c in conditions if c.kind == Kind.RANDOM):
        for stage, alive in enumerate(pattern.alive):
            if alive != everyone:
                state = tuple(alive if s == stage else everyone for s in range(len(pattern.alive)))
                component = PoolCondition(Kind.COMPONENT, grid.scheme, state, pattern.estimator)
                assert component.key in keys


def test_transform_rebuilds_each_stage_from_its_own_children(grid):
    scheme = grid.scheme
    everyone = frozenset(range(3))
    alive = (everyone, frozenset({1}), frozenset({0, 2}))
    condition = PoolCondition(Kind.RANDOM, scheme, alive, Estimator.MEAN)
    transform = condition.transform({"a": 0, "b": 1, "c": 2})
    weight = torch.randn(4, 64, generator=torch.Generator().manual_seed(0))
    assert torch.equal(transform("a", weight), simulate_rebuild(weight, scheme, everyone))
    assert torch.equal(transform("b", weight), simulate_child(weight, scheme, 1))
    expected = simulate_rebuild(weight, scheme, {0, 2}, Estimator.MEAN)
    assert torch.equal(transform("c", weight), expected)
    assert condition.lost == [(1, 0), (1, 2), (2, 1)]
    assert condition.id == "random/mean/lost-1.0+1.2+2.1"
