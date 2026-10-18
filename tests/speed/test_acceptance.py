import pytest
import torch

from syncytium.speed import agreement_curve, greedy_choices, tokens_per_cycle


def agree(*rows):
    return torch.tensor([[c == "1" for c in row] for row in rows])


def test_full_agreement_yields_the_drafts_plus_the_groups_token():
    assert tokens_per_cycle(agree("1" * 12), 3) == 4.0


def test_no_agreement_yields_one_token_per_cycle():
    assert tokens_per_cycle(agree("0" * 7), 4) == 1.0


def test_drafts_run_until_the_first_disagreement():
    # Cycles: "11" + rejection -> 3 tokens, "0" -> 1, "111" + rejection -> 4, then "1" -> 1.
    assert tokens_per_cycle(agree("110011101"), 4) == pytest.approx(9 / 4)


def test_the_curve_rises_with_draft_length_under_full_agreement():
    # 60 positions split evenly into cycles of 2, 3 and 5 tokens.
    curve = agreement_curve(agree("1" * 60), [1, 2, 4])
    assert curve == {1: 2.0, 2: 3.0, 4: 5.0}


def test_draft_length_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        tokens_per_cycle(agree("1"), 0)


def test_greedy_choices_are_the_argmax_of_the_logits(tiny_model):
    batches = torch.randint(0, 256, (2, 9), generator=torch.Generator().manual_seed(0))
    choices = greedy_choices(tiny_model, batches)
    assert choices.shape == (2, 9)
    with torch.no_grad():
        expected = tiny_model(batches).logits.argmax(dim=-1).int()
    assert torch.equal(choices, expected)
