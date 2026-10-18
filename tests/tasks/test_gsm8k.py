import pytest

from syncytium.tasks import gsm8k


def test_prompt_has_eight_worked_examples_and_ends_after_a():
    text = gsm8k.prompt("How many apples?")
    assert text.count("Q: ") == 9
    assert text.count("The answer is") == 8
    assert text.endswith("Q: How many apples?\nA:")


def test_reference_is_the_text_after_the_marker():
    assert gsm8k.reference("He has 3 + 4 = <<3+4=7>>7 apples.\n#### 1,234") == "1,234"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (" 3 + 4 = 7. The answer is 7.", "7"),
        (" She pays $1,200.50 in total. The answer is $1,200.50.", "1200.50"),
        (" The temperature drops to -5 degrees. The answer is -5.", "-5"),
        (" There is no number here.", None),
        (" 12 apples, then 18.", "18"),
    ],
)
def test_extract_takes_the_last_number(answer, expected):
    assert gsm8k.extract(answer) == expected


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [("18", "18", True), ("18.00", "18", True), ("1200", "1,200", True), ("17", "18", False)],
)
def test_answers_are_compared_as_numbers(prediction, gold, expected):
    assert gsm8k.is_correct(prediction, gold) is expected


def test_missing_prediction_is_wrong():
    assert not gsm8k.is_correct(None, "3")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 7. The answer is 7.\n\nQ: next", " 7. The answer is 7."),
        (" 7.Q: x", " 7."),
        (" 7.", " 7."),
    ],
)
def test_truncate_stops_at_the_first_stop_sequence(text, expected):
    assert gsm8k.truncate(text) == expected


@pytest.fixture
def dataset(monkeypatch):
    from datasets import Dataset

    rows = Dataset.from_dict(
        {"question": [f"q{i}" for i in range(10)], "answer": [f"... #### {i}" for i in range(10)]}
    )
    monkeypatch.setattr(gsm8k, "load_dataset", lambda *args, **kwargs: rows)


@pytest.mark.parametrize(
    ("limit", "offset", "expected"),
    [
        (None, 0, list(range(10))),
        (3, 0, [0, 1, 2]),
        (3, 4, [4, 5, 6]),
        (None, 8, [8, 9]),
        (5, 8, [8, 9]),
    ],
)
def test_problems_are_taken_in_order_from_the_offset(dataset, limit, offset, expected):
    problems = gsm8k.load_gsm8k("train", limit, offset)
    assert [p.question for p in problems] == [f"q{i}" for i in expected]
    assert [p.answer for p in problems] == [str(i) for i in expected]
