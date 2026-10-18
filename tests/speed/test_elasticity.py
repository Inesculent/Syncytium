import pytest

from syncytium.speed import elasticity
from syncytium.speed.elasticity import Costs, System, check_h10b, systems

COSTS = Costs(solo=0.04, fused=0.11, large=0.07, hop=0.0003)
N = 20


def rows(predictions, correct):
    return [
        {"index": i, "prediction": p, "correct": c}
        for i, (p, c) in enumerate(zip(predictions, correct, strict=True))
    ]


@pytest.fixture
def items():
    truth = [str(i) for i in range(N)]
    worst = rows(truth[:15] + [str(100 + i) for i in range(5)], [True] * 15 + [False] * 5)
    good = rows(truth, [True] * N)
    return {
        "bf16": good,
        "child/k0": worst,
        "child/k1": good,
        "child/k2": good,
        "child/k3": good,
        "group/k0,1,2,3": good,
        "rtn/b6": rows(truth, [True] * 18 + [False] * 2),
    }


def test_throughputs_follow_the_preregistered_device_accounting(items):
    swarm, split_a, split_b, floor = systems(items, COSTS)
    escalated = 0.25
    assert swarm.throughput == pytest.approx(4 / (2 * 0.04 + escalated * 4 * 0.11))
    assert split_a.throughput == pytest.approx(min(2 / 0.08, 2 / (escalated * (0.07 + 0.0003))))
    assert split_b.throughput == pytest.approx(4 / (0.07 + 3 * 0.0003))
    assert floor.throughput == pytest.approx(4 / 0.04)


def test_accuracies_route_disagreements_to_the_stronger_model(items):
    swarm, split_a, split_b, floor = systems(items, COSTS)
    assert swarm.accuracy == 1.0  # every escalated problem is right in the group
    assert split_a.accuracy == pytest.approx(0.90)  # 6-bit misses two of the five escalated
    assert split_b.accuracy == pytest.approx(0.90)
    assert floor.accuracy == pytest.approx((0.75 + 3) / 4)


def test_h10b_compares_only_with_splits_at_least_as_accurate():
    swarm = System("swarm", 0.86, 20.0)
    floor = System("floor: four lone agents", 0.80, 100.0)
    slower = System("split A: x", 0.86, 15.0)
    less_accurate = System("split B: y", 0.85, 50.0)
    assert check_h10b([swarm, slower, less_accurate, floor]).passed
    faster = System("split B: y", 0.86, 50.0)
    check = check_h10b([swarm, slower, faster, floor])
    assert not check.passed
    assert "split A, split B" in check.detail


def test_confirmation_reports_both_hypotheses(items, tmp_path, monkeypatch):
    import json

    for name, found in items.items():
        path = tmp_path / "items" / "gsm8k" / f"{name.replace('/', '-')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in found), encoding="utf-8")
    monkeypatch.setattr(elasticity, "costs", lambda config: COSTS)
    text = elasticity.confirmation(tmp_path, config=None)
    assert "**H8b: pass.**" in text
    assert "**H10b: " in text
    assert "| swarm | 1.000 |" in text
