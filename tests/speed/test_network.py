import pytest

from syncytium.speed import (
    Network,
    Round,
    Swarm,
    all_reduce_seconds,
    argmax_seconds,
    fused_pass_seconds,
    fused_tokens_per_second,
    solo_seconds,
)

NET = Network(latency=1e-3, bandwidth=1e9, memory_bandwidth=100e9)
SWARM = Swarm(
    rounds=[Round(("a",), 1000), Round(("head",), 50_000, head=True)], child_bytes=4e9, devices=4
)


def test_solo_speed_is_one_weight_read_per_token():
    assert solo_seconds(SWARM, NET) == pytest.approx(0.04)


def test_small_messages_use_the_one_step_exchange():
    # 1000 bf16 values = 16 kbit; direct: 1 ms + 3 * 16 µs, two-step: 2 ms + 1.5 * 16 µs.
    assert all_reduce_seconds(1000, 4, NET) == pytest.approx(1e-3 + 3 * 16e-6)


def test_large_messages_use_reduce_scatter_and_all_gather():
    bits = 10_000_000 * 16
    expected = 2e-3 + 2 * 3 / 4 * bits / 1e9
    assert all_reduce_seconds(10_000_000, 4, NET) == pytest.approx(expected)


def test_argmax_needs_only_a_quarter_of_the_logits_from_each_device():
    bits = 50_000 * 16
    expected = 1e-3 + 3 / 4 * bits / 1e9 + 1e-3 + 3 * 64 / 1e9
    assert argmax_seconds(50_000, 4, NET) == pytest.approx(expected)
    assert argmax_seconds(50_000, 4, NET) < all_reduce_seconds(50_000, 4, NET)


def test_fused_pass_adds_every_round_to_one_weight_read():
    expected = 0.04 + all_reduce_seconds(1000, 4, NET) + argmax_seconds(50_000, 4, NET)
    assert fused_pass_seconds(SWARM, 1, NET) == pytest.approx(expected)
    assert fused_tokens_per_second(SWARM, NET) == pytest.approx(1 / expected)


def test_speculation_pays_drafts_and_one_check_over_k_plus_one_positions():
    cycle = 3 * 0.04 + fused_pass_seconds(SWARM, 4, NET)
    assert fused_tokens_per_second(SWARM, NET, 3, 2.5) == pytest.approx(2.5 / cycle)


def test_speculation_helps_only_when_talking_costs_more_than_reading_weights():
    # A draft costs a full weight read, so drafting pays off only once the rounds dominate.
    many = Swarm(rounds=[Round(("l",), 1000)] * 130, child_bytes=4e9, devices=4)
    slow = Network(latency=1e-3, bandwidth=10e9, memory_bandwidth=100e9)
    assert fused_tokens_per_second(many, slow, 4, 4.0) > fused_tokens_per_second(many, slow)
    fast = Network(latency=1e-6, bandwidth=100e9, memory_bandwidth=100e9)
    assert fused_tokens_per_second(many, fast, 4, 4.0) < fused_tokens_per_second(many, fast)
