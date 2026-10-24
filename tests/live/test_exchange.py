import socket
import threading
import time

import pytest
import torch

from syncytium.live.exchange import DroppedError, Exchange, mean_in_order
from syncytium.live.wire import Endpoint, Link, Message, decode, encode

TIMEOUT = 0.4
ROUNDS = 6


class CrashError(Exception):
    """Raised inside a member's thread to stop it, as a crash would."""


def value(member, r):
    return torch.full((3,), float(10 * member + r), dtype=torch.bfloat16)


def run(members, crash=None, exchange_type=Exchange):
    """Run ``ROUNDS`` rounds on every member in its own thread; return each member's results.

    ``crash`` maps a member to the round at whose start it dies without contributing.
    """
    crash = crash or {}
    endpoints = {m: Endpoint(m, host="127.0.0.1") for m in members}
    addresses = {m: ("127.0.0.1", e._server.getsockname()[1]) for m, e in endpoints.items()}
    results = {m: {} for m in members}
    exchanges = {}

    def member(m):
        exchange = exchange_type(endpoints[m], addresses, members, timeout=TIMEOUT)
        exchanges[m] = exchange
        try:
            for r in range(1, ROUNDS + 1):
                if crash.get(m) == r:
                    raise CrashError
                results[m][r] = exchange.average(value(m, r))
        except CrashError:
            endpoints[m].close()

    threads = [threading.Thread(target=member, args=(m,)) for m in members]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for e in endpoints.values():
        e.close()
    return results, exchanges


def assert_consistent(results, survivors):
    for r in range(1, ROUNDS + 1):
        values = [results[m][r] for m in survivors]
        assert all(torch.equal(v, values[0]) for v in values), f"members disagree in round {r}"


def test_members_get_the_ordered_mean_every_round():
    results, _ = run([0, 1, 2, 3])
    assert_consistent(results, [0, 1, 2, 3])
    for r in range(1, ROUNDS + 1):
        expected = mean_in_order({m: value(m, r) for m in range(4)})
        assert torch.equal(results[2][r], expected)


def test_a_member_that_dies_is_dropped_and_the_rest_continue():
    results, exchanges = run([0, 1, 2, 3], crash={2: 3})
    assert_consistent(results, [0, 1, 3])
    assert torch.equal(results[0][2], mean_in_order({m: value(m, 2) for m in range(4)}))
    assert torch.equal(results[0][3], mean_in_order({m: value(m, 3) for m in (0, 1, 3)}))
    assert exchanges[0].members == [0, 1, 3]
    assert any(
        e["event"] == "dropped" and e["member"] == 2 and e["round"] == 3
        for e in exchanges[0].events
    )


def test_a_dead_leader_is_replaced():
    results, exchanges = run([0, 1, 2, 3], crash={0: 3})
    assert_consistent(results, [1, 2, 3])
    assert torch.equal(results[1][3], mean_in_order({m: value(m, 3) for m in (1, 2, 3)}))
    assert exchanges[1].members == [1, 2, 3]
    assert any(e["event"] == "leader" for e in exchanges[1].events)


class LeaderDiesMidBroadcast(Exchange):
    """A leader that sends round 3's result to member 1 only, then crashes."""

    def _broadcast(self, r, result, targets):
        if self.me == 0 and r == 3:
            message = Message("result", self.me, r, {"members": self.members}, result)
            self._send(1, message)
            raise CrashError
        super()._broadcast(r, result, targets)


def test_a_leader_dying_mid_broadcast_leaves_every_member_with_the_same_result():
    results, _ = run([0, 1, 2, 3], exchange_type=LeaderDiesMidBroadcast)
    assert_consistent(results, [1, 2, 3])
    # Member 1 already had round 3's result, which included the old leader; everyone adopts it.
    assert torch.equal(results[2][3], mean_in_order({m: value(m, 3) for m in range(4)}))
    assert torch.equal(results[2][4], mean_in_order({m: value(m, 4) for m in (1, 2, 3)}))


def test_a_new_leader_can_take_over_in_the_first_round():
    results, exchanges = run([0, 1, 2, 3], crash={0: 1})
    assert_consistent(results, [1, 2, 3])
    assert torch.equal(results[1][1], mean_in_order({m: value(m, 1) for m in (1, 2, 3)}))
    assert exchanges[1].members == [1, 2, 3]


class SlowState(Exchange):
    """Members contribute once the leader is surely down; member 2 sends its state late."""

    def _send(self, member, message):
        if message.kind == "contrib" and message.round == 1:
            time.sleep(0.2)  # member 0 crashes at once, so every contribution is refused
        if self.me == 2 and message.kind == "state":
            time.sleep(1.5 * self._patience)  # later than one patience after the takeover
        return super()._send(member, message)


def test_a_new_leader_waits_for_a_member_slow_to_send_its_state():
    results, exchanges = run([0, 1, 2, 3], crash={0: 1}, exchange_type=SlowState)
    assert_consistent(results, [1, 2, 3])
    assert torch.equal(results[2][1], mean_in_order({m: value(m, 1) for m in (1, 2, 3)}))
    assert exchanges[2].members == [1, 2, 3]


class SlowMember(Exchange):
    """Member 3 sends round 2's contribution to the first request after the leader gave up."""

    def _send(self, member, message):
        if (self.me, self.tag, message.kind, message.round) == (3, 0, "contrib", 2):
            time.sleep(1.5 * TIMEOUT)
        return super()._send(member, message)


def two_requests(m, endpoint, addresses, members, results, left):
    """Member ``m``'s part in a request where member 3 is late, then in the next request."""
    first = SlowMember(endpoint, addresses, members, TIMEOUT, tag=0)
    try:
        for r in (1, 2, 3):
            first.average(value(m, r))
    except DroppedError:
        left[m] = first.events
    # The next request starts with everyone. Member 3's late message from the first one has
    # the same round number, but must not be taken for its contribution to this one.
    second = Exchange(endpoint, addresses, members, TIMEOUT, tag=1)
    for r in range(1, ROUNDS + 1):
        results[m][r] = second.average(value(m, r) + 100)


def test_a_late_member_is_dropped_leaves_and_cannot_touch_the_next_request():
    members = [0, 1, 2, 3]
    endpoints = {m: Endpoint(m, host="127.0.0.1") for m in members}
    addresses = {m: ("127.0.0.1", e._server.getsockname()[1]) for m, e in endpoints.items()}
    results, left = {m: {} for m in members}, {}
    threads = [
        threading.Thread(
            target=two_requests, args=(m, endpoints[m], addresses, members, results, left)
        )
        for m in members
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for e in endpoints.values():
        e.close()
    assert list(left) == [3] and left[3][-1]["event"] == "left"
    assert_consistent(results, members)
    for r in range(1, ROUNDS + 1):
        assert torch.equal(results[3][r], mean_in_order({m: value(m, r) + 100 for m in members}))


@pytest.mark.parametrize(
    "tensor",
    [torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), torch.zeros(0), torch.tensor([3.5])],
)
def test_messages_round_trip(tensor):
    message = Message("contrib", 2, 7, {"members": [0, 2]}, tensor)
    data = encode(message)
    header_length = int.from_bytes(data[:4], "big")
    payload_length = int.from_bytes(data[4:12], "big")
    back = decode(data[12 : 12 + header_length], data[12 + header_length :])
    assert (back.kind, back.sender, back.round, back.info) == ("contrib", 2, 7, {"members": [0, 2]})
    assert back.tensor.dtype == tensor.dtype
    assert torch.equal(back.tensor, tensor)
    assert payload_length == tensor.numel() * tensor.element_size()


def test_an_emulated_link_serializes_each_end_and_adds_latency_once():
    link = Link(latency=0.0002, bandwidth=1e9)
    assert link.serialize(125_000) == pytest.approx(0.001)
    assert Link().serialize(10**9) == 0.0


def test_rounds_on_an_emulated_link_cost_what_the_model_says():
    from syncytium.speed.network import Network, leader_reduce_seconds

    link = Link(latency=0.002, bandwidth=1e8)
    values = 50_000  # 100 kB of bf16 per vector, 8 ms on this link
    members = [0, 1, 2, 3]
    rounds = 6
    endpoints = {m: Endpoint(m, host="127.0.0.1", link=link) for m in members}
    addresses = {m: ("127.0.0.1", e._server.getsockname()[1]) for m, e in endpoints.items()}
    elapsed = {}

    def member(m):
        exchange = Exchange(endpoints[m], addresses, members, timeout=5)
        exchange.average(torch.zeros(values, dtype=torch.bfloat16))  # warm up the connections
        # Steady-state rounds, as check (g) measures them: a single round would also count
        # the skew between members left over from the previous one.
        start = time.monotonic()
        for _ in range(rounds):
            exchange.average(torch.full((values,), float(m), dtype=torch.bfloat16))
        elapsed[m] = (time.monotonic() - start) / rounds

    threads = [threading.Thread(target=member, args=(m,)) for m in members]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for e in endpoints.values():
        e.close()
    predicted = leader_reduce_seconds(values, 4, Network(0.002, 1e8, 1e11))
    assert predicted / 2 < max(elapsed.values()) < predicted * 2


def test_direct_peers_skip_the_emulated_link():
    slow = Link(latency=0.5, bandwidth=1e6)  # a 1 MB message takes 8 s
    sender, receiver = (Endpoint(k, host="127.0.0.1", link=slow) for k in (1, 2))
    sender.direct, receiver.direct = frozenset({2}), frozenset({1})  # each names the other
    bulk = Message("bulk", 1, tensor=torch.zeros(500_000, dtype=torch.bfloat16))
    address = ("127.0.0.1", receiver._server.getsockname()[1])
    start = time.monotonic()
    assert sender.send(2, address, bulk)
    assert receiver.inbox.take(lambda m: m.kind == "bulk", 2) is not None
    assert time.monotonic() - start < 2

    # Without the exemption the same message waits for the link.
    sender.direct = receiver.direct = frozenset()
    thread = threading.Thread(target=sender.send, args=(2, address, bulk), daemon=True)
    thread.start()
    assert receiver.inbox.take(lambda m: m.kind == "bulk", 1) is None
    for endpoint in (sender, receiver):
        endpoint.close()


def test_a_large_send_to_one_peer_does_not_hold_up_another():
    # A peer that accepts the connection but never reads, so a large send to it stays blocked.
    stuck = socket.create_server(("127.0.0.1", 0))
    sender, other = (Endpoint(k, host="127.0.0.1") for k in (1, 2))
    bulk = Message("bulk", 1, tensor=torch.zeros(64_000_000, dtype=torch.uint8))
    threading.Thread(target=sender.send, args=(3, stuck.getsockname(), bulk), daemon=True).start()
    time.sleep(0.3)
    start = time.monotonic()
    assert sender.send(2, ("127.0.0.1", other._server.getsockname()[1]), Message("beat", 1))
    assert other.inbox.take(lambda m: m.kind == "beat", 2) is not None
    assert time.monotonic() - start < 1
    for closing in (sender, other, stuck):
        closing.close()
