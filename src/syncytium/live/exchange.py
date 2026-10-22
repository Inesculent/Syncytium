"""Averaging layer outputs among the live members of a fused group (H18).

Every member sends its output for a round to the leader, the lowest live member. The leader
sums the contributions in member order in float32, divides, and sends the average back with the
current member list. Summing in a fixed order makes the result deterministic, so the same
members and the same children always give the same bits.

Members die. The leader drops any member whose contribution misses the deadline and finishes
the round with the rest. A member that gets no result in time treats the leader as dead; the
next lowest member takes over. Members can then be at most one round apart, because a round's
result needs every contribution to the next. The new leader collects each member's state,
fetches the result it missed from a member that has it, hands results to members that lack
them, and carries on, so every member always holds the same value for every round. A member
the leader dropped while it was still alive is told so with the round's result, and leaves the
request instead of taking over.

Every message carries its request's tag. Rounds are numbered from 1 in every request, so without
it a late message from one request could be taken for the same round of a later one.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from syncytium.live.wire import Endpoint, Message

CONTRIB, RESULT, STATE, WANT = "contrib", "result", "state", "want"
KINDS = (CONTRIB, RESULT, STATE, WANT)
# Bits per second the real network is assumed to move at least. Waits allow for it on top of
# any emulated link, so a large round (a probe's logits) is not taken for a dead member.
NETWORK_FLOOR = 1e9


class DroppedError(Exception):
    """Raised on a member the leader dropped from the group; it takes no further part."""


class Exchange:
    """Averaging for one fused request among ``members``.

    Args:
        endpoint: This device's endpoint; its index is this member's index.
        addresses: Address of every member, by member index.
        members: The members of the group when the request starts.
        timeout: Seconds the leader waits for a contribution. Members wait longer for a
            result, so a leader waiting out a dead member is not itself taken for dead. Every
            wait also allows the time the network needs to move the round's vectors through
            the leader: the emulated link, if any, plus the real one at
            :data:`NETWORK_FLOOR`.
        tag: The request this exchange belongs to; later requests have larger tags.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        addresses: Mapping[int, tuple[str, int]],
        members: Sequence[int],
        timeout: float = 5.0,
        tag: int = 0,
    ) -> None:
        self.endpoint, self.addresses = endpoint, dict(addresses)
        self.tag = tag
        self.me = endpoint.index
        self.members = sorted(members)
        self.timeout = timeout
        self.round = 0
        self.events: list[dict[str, Any]] = []
        # One (values, live members, seconds) per round, for the communication check (g).
        self.timing: list[tuple[int, int, float]] = []
        self._slack = 0.0
        self._results: dict[int, Tensor] = {}
        self._buffer: dict[int, dict[int, Tensor]] = {}
        # Whatever earlier requests left behind is never needed again.
        endpoint.inbox.discard(lambda msg: msg.kind in KINDS and self._tag(msg) < tag)

    @property
    def _patience(self) -> float:
        """How long a member waits on the leader before taking it for dead."""
        return 2.5 * self.timeout + 2 * self._slack

    @property
    def leader(self) -> int:
        """The lowest live member."""
        return self.members[0]

    def average(self, x: Tensor) -> Tensor:
        """The mean of every live member's ``x`` for the next round, the same on every member."""
        self.round += 1
        r = self.round
        start, live = time.monotonic(), len(self.members)
        local = x.detach().to("cpu")
        size = local.numel() * local.element_size()
        moving = self.endpoint.link.serialize(size) + size * 8 / NETWORK_FLOOR
        self._slack = 2 * live * moving
        result = self._lead(r, local) if self.leader == self.me else self._follow(r, local)
        self.timing.append((local.numel(), live, time.monotonic() - start))
        self._results[r] = result
        self._results.pop(r - 3, None)
        return result.to(x.device)

    def _lead(self, r: int, local: Tensor) -> Tensor:
        contributions = {self.me: local, **self._buffer.pop(r, {})}
        # Members dropped this round still get its result, which tells them they were.
        targets = [m for m in self.members if m != self.me]
        deadline = time.monotonic() + self.timeout + self._slack
        for member in [m for m in self.members if m not in contributions]:
            message = self._take(CONTRIB, member, deadline - time.monotonic(), r)
            if message is None:
                self._drop(member, r)
            else:
                contributions[member] = message.tensor
        result = mean_in_order({m: contributions[m] for m in self.members if m in contributions})
        self._broadcast(r, result, targets)
        return result

    def _follow(self, r: int, local: Tensor) -> Tensor:
        sent = self._send(self.leader, Message(CONTRIB, self.me, r, tensor=local))
        result = self._await_result(r) if sent else None
        while result is None:
            self._drop(self.leader, r)
            if self.leader == self.me:
                return self._take_over(r, local)
            sent = self._send(self.leader, Message(STATE, self.me, r, tensor=local))
            result = self._await_result(r) if sent else None
        return result

    def _await_result(self, r: int) -> Tensor | None:
        """Wait for round ``r``'s result from the leader, answering its requests meanwhile.

        Raises:
            DroppedError: If the result's members leave this one out.
        """
        leader = self.leader
        deadline = time.monotonic() + self._patience
        while (left := deadline - time.monotonic()) > 0:
            message = self.endpoint.inbox.take(
                lambda msg: msg.sender == leader and msg.kind in (RESULT, WANT) and self._ours(msg),
                left,
            )
            if message is None:
                break
            if message.kind == WANT:
                reply = Message(RESULT, self.me, message.round, tensor=self._results[message.round])
                self._send(leader, reply)
                continue
            if message.round == r:
                if self.me not in message.info["members"]:
                    self._log("left", r, self.me)
                    raise DroppedError(f"member {self.me} was dropped in round {r}")
                self.members = message.info["members"]
                return message.tensor
        return None

    def _take_over(self, p: int, local: Tensor) -> Tensor:
        """Become leader while waiting for round ``p``'s result; reconcile and continue."""
        self._log("leader", p, self.me)
        states = self._collect_states()
        ahead = [m for m, s in states.items() if s.round == p + 1]
        behind = [m for m, s in states.items() if s.round == p - 1]
        if ahead:
            result = self._fetch(ahead[0], p)
            laggards = [m for m, s in states.items() if s.round == p]
            self._buffer[p + 1] = {m: states[m].tensor for m in ahead}
            self._broadcast(p, result, laggards)
            return result
        if behind:
            self._broadcast(p - 1, self._results[p - 1], behind)
        self._buffer[p] = {m: s.tensor for m, s in states.items() if s.round == p}
        return self._lead(p, local)

    def _collect_states(self) -> dict[int, Message]:
        states = {}
        deadline = time.monotonic() + self._patience
        for member in [m for m in self.members if m != self.me]:
            message = self._take(STATE, member, deadline - time.monotonic())
            if message is None:
                self._drop(member, self.round)
            else:
                states[member] = message
        return states

    def _fetch(self, member: int, r: int) -> Tensor:
        """Round ``r``'s result from ``member``, which already has it."""
        self._send(member, Message(WANT, self.me, r))
        message = self._take(RESULT, member, self._patience, r)
        if message is None:
            raise RuntimeError(f"member {member} had round {r}'s result but did not send it")
        return message.tensor

    def _broadcast(self, r: int, result: Tensor, targets: Sequence[int]) -> None:
        for member in targets:
            message = Message(RESULT, self.me, r, {"members": self.members}, result)
            if not self._send(member, message):
                self._drop(member, r)

    def _send(self, member: int, message: Message) -> bool:
        message.info["exchange"] = self.tag
        return self.endpoint.send(member, self.addresses[member], message)

    def _take(self, kind: str, member: int, wait: float, r: int | None = None) -> Message | None:
        """This request's next ``kind`` message from ``member`` (for round ``r`` if given)."""
        return self.endpoint.inbox.take(
            lambda msg: (
                msg.kind == kind
                and msg.sender == member
                and (r is None or msg.round == r)
                and self._ours(msg)
            ),
            max(wait, 0.0),
        )

    def _ours(self, message: Message) -> bool:
        return self._tag(message) == self.tag

    @staticmethod
    def _tag(message: Message) -> int:
        return message.info.get("exchange", -1)

    def _drop(self, member: int, r: int) -> None:
        if member in self.members:
            self.members = [m for m in self.members if m != member]
            self._log("dropped", r, member)

    def _log(self, event: str, r: int, member: int) -> None:
        self.events.append({"event": event, "round": r, "member": member, "time": time.time()})


def mean_in_order(contributions: Mapping[int, Tensor]) -> Tensor:
    """The leader's average: a float32 sum in member order, divided, cast back."""
    ordered = [contributions[m] for m in sorted(contributions)]
    return (sum(t.float() for t in ordered) / len(ordered)).to(ordered[0].dtype)
