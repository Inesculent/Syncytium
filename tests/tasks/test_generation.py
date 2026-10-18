import math

import pytest
import torch

from syncytium.tasks import Decoding, continue_after_switch, generate

DECODING = Decoding(max_new_tokens=12, stops=("\n\n",))


@pytest.fixture
def prompts():
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(2, 256, (n,), generator=generator).tolist() for n in (5, 9, 13)]


def test_batched_answers_equal_answers_generated_one_at_a_time(tiny_model, tokenizer, prompts):
    batched = generate(tiny_model, tokenizer, prompts, DECODING)
    alone = [generate(tiny_model, tokenizer, [p], DECODING)[0] for p in prompts]
    assert [c.tokens for c in batched] == [c.tokens for c in alone]
    for together, single in zip(batched, alone, strict=True):
        assert together.entropy == pytest.approx(single.entropy, rel=1e-4)


def test_entropy_is_finite_and_bounded_by_the_vocabulary(tiny_model, tokenizer, prompts):
    for completion in generate(tiny_model, tokenizer, prompts, DECODING):
        assert 0 <= completion.entropy <= math.log(256) + 1e-6


def test_answers_end_before_the_first_stop(tiny_model, tokenizer, prompts):
    stops = ("a", "e", "i", "o", "u")  # frequent bytes, so random answers stop early
    for completion in generate(tiny_model, tokenizer, prompts, Decoding(12, stops)):
        assert not any(stop in completion.text for stop in stops)
        assert tokenizer.decode(completion.tokens) == completion.text


def test_continuing_without_a_switch_reproduces_the_answer(tiny_model, tokenizer, prompts):
    reference = generate(tiny_model, tokenizer, prompts, DECODING)
    prefixes = [c.tokens[: len(c.tokens) // 2] for c in reference]
    continued = continue_after_switch(
        tiny_model, tokenizer, prompts, prefixes, lambda: None, DECODING
    )
    assert [c.tokens for c in continued] == [c.tokens for c in reference]


def test_switched_weights_write_the_rest_of_the_answer(tiny_model, tokenizer, prompts):
    reference = generate(tiny_model, tokenizer, prompts, DECODING)
    prefixes = [c.tokens[:4] for c in reference]

    @torch.no_grad()
    def switch():
        for parameter in tiny_model.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))

    continued = continue_after_switch(tiny_model, tokenizer, prompts, prefixes, switch, DECODING)
    for answer, prefix, before in zip(continued, prefixes, reference, strict=True):
        # The prefix and the token chosen before the switch are kept; the rest may change.
        assert answer.tokens[:5] == before.tokens[:5]
        assert answer.tokens[: len(prefix)] == prefix
        assert len(answer.tokens) <= DECODING.max_new_tokens
    assert [c.tokens for c in continued] != [c.tokens for c in reference]
