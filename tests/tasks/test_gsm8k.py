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
