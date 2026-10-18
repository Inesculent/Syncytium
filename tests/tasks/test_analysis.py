import math

import pytest

from syncytium.tasks import analysis


def rows(correct, entropy=None):
    entropy = entropy or [0.0] * len(correct)
    return [
        {"index": i, "correct": bool(c), "entropy": e}
        for i, (c, e) in enumerate(zip(correct, entropy, strict=True))
    ]


def test_mcnemar_is_an_exact_binomial_test_on_discordant_pairs():
    assert analysis.mcnemar([True] * 10, [False] * 10) == pytest.approx(2 / 2**10)
    assert analysis.mcnemar([True, False], [False, True]) == 1.0
    assert analysis.mcnemar([True, True], [True, True]) == 1.0


def test_escalation_hands_the_most_uncertain_answers_to_the_group():
    entropy = [0.1, 0.9, 0.5, 0.8]
    lone = [True, False, True, False]
    fused = [True, True, False, True]
    assert analysis.escalate(entropy, lone, fused, 0.0) == 0.5
    assert analysis.escalate(entropy, lone, fused, 0.5) == 1.0  # items 1 and 3
    assert analysis.escalate(entropy, lone, fused, 1.0) == 0.75


def test_noninferiority_bound():
    assert analysis.noninferiority_bound([0.0] * 10) == 0.0
    bound = analysis.noninferiority_bound([1.0, -1.0] * 50)
    assert bound == pytest.approx(-1.6448536 * 1.0050378 / 10, rel=1e-6)


def items(bf16, children, groups, drops=None):
    found = {"bf16": rows(bf16)}
    found.update({f"child/k{k}": rows(*c) for k, c in enumerate(children)})
    found.update({f"group/k{','.join(map(str, g))}": rows(c) for g, c in groups.items()})
    found.update({f"drop/k{k}": rows(c) for k, c in (drops or {}).items()})
    return found


def test_h7_needs_half_the_gap_three_points_and_significance():
    n = 100
    child = [i < 60 for i in range(n)]
    found = items(
        bf16=[i < 80 for i in range(n)],
        children=[(child,), ([i < 70 for i in range(n)],)],
        groups={(0, 1): [i < 75 for i in range(n)]},
    )
    check = analysis.check_h7(found)
    assert check.passed  # 15 of a 20-point gap, p = 2^-14
    found["group/k0,1"] = rows([i < 65 for i in range(n)])
    assert not analysis.check_h7(found).passed  # 5 of 20 points is 25% of the gap


def test_h7_fails_when_the_lone_child_already_matches_bf16():
    same = [True, False] * 50
    found = items(bf16=same, children=[(same,), (same,)], groups={(0, 1): same})
    check = analysis.check_h7(found)
    assert not check.passed
    assert "n/a" in check.detail


def test_h8_judges_the_worst_child_and_reports_the_others():
    n = 100
    lone = [i >= 40 for i in range(n)]  # wrong on the first 40
    entropy = [1.0 if i < 30 else 0.0 for i in range(n)]  # uncertain exactly where it errs
    found = items(
        bf16=[True] * n,
        children=[(lone, entropy), ([True] * n,)],
        groups={(0, 1): [i >= 10 for i in range(n)]},
    )
    checks = analysis.check_h8(found)
    assert [c.hypothesis for c in checks] == ["H8", "H8 (child 1, reference)"]
    assert checks[0].passed  # escalating 30% recovers 20 of the 30-point gain


def test_h11_checks_the_ordering_by_group_size():
    n = 10
    found = items(
        bf16=[True] * n,
        children=[
            ([i < 5 for i in range(n)],),
            ([i < 6 for i in range(n)],),
            ([i < 5 for i in range(n)],),
        ],
        groups={
            (0, 1, 2): [i < 9 for i in range(n)],
            (0, 1): [i < 7 for i in range(n)],
            (0, 2): [i < 7 for i in range(n)],
            (1, 2): [i < 8 for i in range(n)],
        },
    )
    assert analysis.check_h11(found).passed
    found["group/k1,2"] = rows([i < 5 for i in range(n)])  # worst pair no better than best child
    assert not analysis.check_h11(found).passed


def test_h13_is_non_inferiority_with_a_two_point_margin():
    n = 400
    base = [i % 2 == 0 for i in range(n)]
    found = items(
        bf16=base,
        children=[(base,), (base,)],
        groups={(0, 1): base, (0,): base, (1,): base},
        drops={0: base, 1: base},
    )
    assert analysis.check_h13(found).passed
    worse = [c and i % 10 != 0 for i, c in enumerate(base)]  # loses 10 points
    found["drop/k0"] = found["drop/k1"] = rows(worse)
    assert not analysis.check_h13(found).passed


def test_report_lists_accuracies_and_checks(tmp_path):
    for name, correct in [
        ("bf16", [True, True]),
        ("child-k0", [False, True]),
        ("child-k1", [True, False]),
        ("group-k0,1", [True, True]),
    ]:
        path = tmp_path / "items" / "gsm8k" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                f'{{"index": {i}, "correct": {str(c).lower()}, "entropy": 0.0}}\n'
                for i, c in enumerate(correct)
            ),
            encoding="utf-8",
        )
    text = analysis.report(tmp_path)
    assert "| group/k0,1 | 1.0000 | 2/2 |" in text
    assert "**H7: fail.**" in text
    assert not math.isnan(analysis.accuracy(analysis.load_items(tmp_path, "gsm8k")["bf16"]))
