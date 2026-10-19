import json
from itertools import combinations

import pytest

from syncytium.speed import survival
from syncytium.speed.survival import (
    LARGE,
    Point,
    Speeds,
    check_h12b,
    neighbour_restorable,
    pipelines_answer,
    point,
    probability,
    residue_restorable,
    states,
    swarm_answer,
    throughput,
)


def rows(fraction, n=100):
    return [{"index": i, "correct": i < fraction * n} for i in range(n)]


def items(child=0.8, pair=0.84, trio=0.85, full=0.86, large=0.86):
    found = {"bf16": rows(0.86), LARGE: rows(large)}
    for k in range(4):
        found[f"child/k{k}"] = rows(child)
    for size, value in ((2, pair), (3, trio), (4, full)):
        for group in combinations(range(4), size):
            found[f"group/k{','.join(map(str, group))}"] = rows(value)
    return found


@pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
def test_perfect_systems_reduce_to_their_availability(p):
    perfect = items(1, 1, 1, 1, 1)
    swarm = point(swarm_answer(perfect), p)
    pipes = point(pipelines_answer(perfect), p)
    assert swarm.answered == pytest.approx(1 - p**4)
    assert swarm.expected == pytest.approx(1 - p**4)
    assert pipes.answered == pytest.approx((1 - p * p) ** 2)
    assert pipes.accuracy_if_answered == pytest.approx(1.0)


def test_the_swarm_weights_each_state_by_its_live_members():
    p = 0.2
    found = items()
    q = 1 - p
    expected = 4 * p**3 * q * 0.8 + 6 * p**2 * q**2 * 0.84 + 4 * p * q**3 * 0.85 + q**4 * 0.86
    assert point(swarm_answer(found), p).expected == pytest.approx(expected)


def test_restoring_full_health():
    p, q = 0.1, 0.9
    neighbour = q**4 + 4 * p * q**3 + 2 * p**2 * q**2  # none, any one, or an opposite pair
    assert probability(neighbour_restorable, p) == pytest.approx(neighbour)
    assert probability(residue_restorable(1), p) == pytest.approx(q**4 + 4 * p * q**3)
    assert probability(residue_restorable(3), p) == pytest.approx(1 - p**4)  # any survivor
    assert not neighbour_restorable((False, False, False, False))
    assert len(states()) == 16


def test_h12b_needs_a_win_at_every_pass_rate():
    def curve(values):
        return [Point(r, 1.0, v) for r, v in zip((0.05, 0.1, 0.2, 0.3), values, strict=True)]

    pipes = curve([0.86, 0.85, 0.80, 0.70])
    assert check_h12b(curve([0.80, 0.86, 0.85, 0.84]), pipes).passed  # 0.05 is not judged
    assert not check_h12b(curve([0.90, 0.84, 0.85, 0.84]), pipes).passed


def test_throughput_counts_complete_pipelines_and_the_fused_group_size():
    speed = Speeds(fused={1: 25.0, 2: 15.0, 3: 11.0, 4: 9.0}, pipeline=27.0)
    found = items(1, 1, 1, 1, 1)
    swarm, pipes = throughput(found, speed, rate=0.0)
    assert swarm == pytest.approx(9.0)  # every member alive: one fused stream
    assert pipes == pytest.approx(2 * 27.0)  # two complete pipelines


def test_report_states_the_verdict(tmp_path, monkeypatch):
    for name, found in items().items():
        path = tmp_path / "items" / "gsm8k" / f"{name.replace('/', '-')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in found), encoding="utf-8")
    monkeypatch.setattr(
        survival, "speeds", lambda config: Speeds({1: 25.0, 2: 15.0, 3: 11.0, 4: 9.0}, 27.0)
    )
    text = survival.survival(tmp_path, config=None)
    assert "**H12b: pass.**" in text
    assert "| 0.3 |" in text
