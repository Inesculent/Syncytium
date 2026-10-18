import pytest

from syncytium.tasks.disagreement import agree, check_h8b, policies, policy


def answer(prediction, correct):
    return {"prediction": prediction, "correct": correct}


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("18", "18", True),
        ("18", "18.0", True),
        ("18", "17", False),
        (None, "18", False),
        (None, None, False),
    ],
)
def test_answers_agree_only_on_the_same_number(first, second, expected):
    assert agree(answer(first, False), answer(second, False)) is expected


def test_disagreements_take_the_stronger_answer():
    requester = [answer("1", True), answer("2", False), answer(None, False), answer("4", True)]
    partner = [answer("1", True), answer("3", False), answer(None, False), answer("5", False)]
    stronger = [answer("9", False), answer("2", True), answer("3", True), answer("6", False)]
    result = policy(requester, partner, stronger, 7)
    assert result.partner == 7
    assert result.escalated == 0.75  # problems 1, 2 and 3
    assert result.accuracy == 0.75  # kept 1 (right), escalated 2 (right), 3 (right), 4 (wrong)


def rows(predictions, correct):
    return [
        {"index": i, **answer(p, c)}
        for i, (p, c) in enumerate(zip(predictions, correct, strict=True))
    ]


def items(lone, fused):
    found = {f"child/k{k}": rows(*child) for k, child in enumerate(lone)}
    found["group/k0,1,2"] = rows(*fused)
    found["bf16"] = rows(*fused)
    return found


N = 20


def test_h8b_passes_when_disagreement_finds_the_mistakes():
    truth = [str(i) for i in range(N)]
    wrong = [str(100 + i) for i in range(N)]
    worst = (truth[:15] + wrong[15:], [True] * 15 + [False] * 5)  # 75%
    other = (truth, [True] * N)  # disagrees exactly where the worst child errs
    fused = (truth, [True] * N)
    found = items([worst, other, other], fused)
    verdict, *lines = check_h8b(found)
    assert verdict.passed
    assert [line.hypothesis for line in lines] == ["H8b, partner 1", "H8b, partner 2"]
    assert all(p.escalated == 0.25 for p in policies(found, "group/k0,1,2"))


def test_h8b_fails_when_too_much_is_escalated():
    truth = [str(i) for i in range(N)]
    worst = ([str(100 + i) for i in range(N)], [False] * N)  # never agrees with anyone
    other = (truth, [True] * N)
    verdict, *_ = check_h8b(items([worst, other, other], (truth, [True] * N)))
    assert not verdict.passed  # recovers everything, but escalates 100%


def test_h8b_fails_without_a_three_point_gain():
    truth = [str(i) for i in range(N)]
    child = (truth, [True] * N)
    verdict, *_ = check_h8b(items([child, child, child], (truth, [True] * N)))
    assert not verdict.passed
