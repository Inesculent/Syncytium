import pytest
import torch

from syncytium.tasks import mmlu


def question(subject="high_school_physics", text="What is 1 + 1?", answer=1):
    return mmlu.Question(subject, text, ("1", "2", "3", "4"), answer)


def test_prompt_lists_choices_and_ends_before_the_letter():
    text = mmlu.prompt(question(), {})
    assert text.startswith(
        "The following are multiple choice questions (with answers) about high school physics.\n\n"
    )
    assert text.endswith("What is 1 + 1?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:")


def test_prompt_uses_five_shots_and_never_the_question_itself():
    target = question(text="target")
    shots = mmlu.shots_by_subject([target, *(question(text=f"shot {i}") for i in range(6))])
    text = mmlu.prompt(target, shots)
    assert text.count("Answer: B") == 5
    assert text.count("target") == 1
    assert "shot 5" not in text


def test_letters_must_be_single_tokens(tokenizer):
    assert mmlu.letter_ids(tokenizer) == [ord(c) for c in "ABCD"]

    class Splitting:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(ValueError, match="single tokens"):
        mmlu.letter_ids(Splitting())


def test_batched_predictions_equal_predictions_one_at_a_time(tiny_model, tokenizer):
    generator = torch.Generator().manual_seed(2)
    prompts = [torch.randint(2, 256, (n,), generator=generator).tolist() for n in (4, 11, 7)]
    letters = mmlu.letter_ids(tokenizer)
    together = mmlu.predict(tiny_model, prompts, letters, tokenizer.pad_token_id)
    alone = [mmlu.predict(tiny_model, [p], letters, tokenizer.pad_token_id)[0] for p in prompts]
    assert together == alone
    assert all(0 <= p < 4 for p in together)
