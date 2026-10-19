"""The controller of the live swarm (``docs/preregistration-demo.md``).

It tracks which devices are alive from their heartbeats, admits a device's child only if its
shard matches the manifest's hash, routes requests as H8b does (the two lowest live members
answer alone and every live member fuses when their final numbers differ), runs probes, kills
and destroys devices on the scenario's schedule, and orchestrates regrowth: it picks the
regrowth data that can rebuild the loss (refusing neighbour bits for adjacent losses), tells
each new device which survivors hold which files, and admits the regrown child by hash.

Every event goes to a JSON-lines log, from which the timeline figure and the H18 checks are
computed. The controller itself is never failed.
"""

import hashlib
import itertools
import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from syncytium.live.device import CONTROLLER, HEARTBEAT
from syncytium.live.wire import Endpoint, Message
from syncytium.shards import (
    MANIFEST_NAME,
    SCALES_NAME,
    Manifest,
    Redundancy,
    RegrowthManifest,
    ShardError,
    shard_name,
)
from syncytium.shards.regrow import check_losses, regrowth_manifest_name, regrowth_name
from syncytium.tasks import gsm8k

DEAD_AFTER = 4 * HEARTBEAT  # silence after which a device is taken for dead
WAIT = 600.0  # longest wait for any device's reply, in seconds


@dataclass
class DeviceState:
    """What the controller knows of one device.

    Attributes:
        address: Where it listens.
        last_beat: When its last heartbeat arrived (monotonic clock).
        alive: Whether it is taken for alive.
        ready: Whether its child is loaded and admitted.
        peak: Largest GPU memory it has reported, in bytes.
    """

    address: tuple[str, int]
    last_beat: float
    alive: bool = True
    ready: bool = False
    peak: int = 0
    killed_at: float | None = field(default=None, repr=False)


class Controller:
    """Runs the swarm's control plane.

    Args:
        endpoint: The controller's endpoint (index :data:`~syncytium.live.device.CONTROLLER`).
        reference: A directory holding the shard set's manifest and regrowth sidecars.
        tokenizer: The model's tokenizer, to encode prompts and decode nothing else.
        log: The JSON-lines event log to write.
        max_new_tokens: Most answer tokens per request.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        reference: Path,
        tokenizer: Any,
        log: Path,
        max_new_tokens: int = 256,
    ) -> None:
        self.endpoint, self.tokenizer = endpoint, tokenizer
        self.manifest = Manifest.load(reference / MANIFEST_NAME)
        self.sidecars = {
            kind: RegrowthManifest.load(reference / regrowth_manifest_name(kind))
            for kind in Redundancy
            if (reference / regrowth_manifest_name(kind)).exists()
        }
        self.max_new_tokens = max_new_tokens
        self.devices: dict[int, DeviceState] = {}
        self._requests = itertools.count()
        self._log_file = log.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        threading.Thread(target=self._monitor, daemon=True).start()

    def live(self) -> list[int]:
        """Members alive with an admitted child, lowest first."""
        with self._lock:
            return sorted(k for k, d in self.devices.items() if d.alive and d.ready)

    def await_devices(self, indices: Sequence[int], timeout: float = WAIT) -> None:
        """Wait until every device in ``indices`` has registered since it was last lost.

        Raises:
            TimeoutError: If one has not registered in time.
        """
        deadline = time.monotonic() + timeout
        while not all(self._registered(k) for k in indices):
            if time.monotonic() > deadline:
                raise TimeoutError(f"devices {list(indices)} did not all register")
            time.sleep(0.05)

    def admit(self, indices: Sequence[int]) -> dict[int, bool]:
        """Ask each device to load its child; admit it only if its shard has the right hash."""
        hashes = self.manifest.shard_sha256
        requests = {k: self._command(k, "load", {"sha256": hashes[k]}) for k in indices}
        admitted = {}
        for k, request in requests.items():
            reply = self._reply(request, k, ("ready", "refused"))
            admitted[k] = reply is not None and reply.kind == "ready"
            with self._lock:
                self.devices[k].ready = admitted[k]
            fields = reply.info if reply is not None else {"sha256": None}
            self.log("admitted" if admitted[k] else "refused", device=k, **fields)
        return admitted

    def serve(self, prompt: str, force: dict[str, int] | None = None) -> dict[str, Any]:
        """Answer one request, routed as in H8b, retrying while members die under it.

        ``force``, if given, makes every live member fuse, and tells member ``force["victim"]``
        to crash after ``force["die_after"]`` answer tokens.
        """
        ids = self.tokenizer.encode(prompt)
        start, record = time.monotonic(), None
        while record is None and (live := self.live()):
            record = self._route(ids, live, force)
        record = record or {"route": "failed", "members": [], "text": None}
        record |= {"seconds": time.monotonic() - start}
        record["prediction"] = gsm8k.extract(record["text"]) if record["text"] else None
        self.log("request", **record)
        return record

    def probe(self, prompts: Sequence[str]) -> dict[str, Any]:
        """Teacher-force ``prompts`` through every live member together.

        Returns the members, the SHA-256 over every prompt's logits, whether every member
        computed the same logits, and per prompt the top-1 tokens and the last positions'
        logits.
        """
        live, results, consistent = self.live(), [], True
        for prompt in prompts:
            info = self._request_info(self.tokenizer.encode(prompt), live)
            request = self._send_all(live, "probe", info)
            replies = self._collect(request, live, "probe")
            consistent &= len({r.info["sha256"] for r in replies.values()}) == 1
            first = replies[min(replies)]
            results.append(
                {"top1": first.info["top1"], "sha256": first.info["sha256"], "tail": first.tensor}
            )
        digest = hashlib.sha256("".join(r["sha256"] for r in results).encode()).hexdigest()
        self.log("probe", members=live, sha256=digest, consistent=consistent)
        return {"members": live, "sha256": digest, "consistent": consistent, "prompts": results}

    def kill(self, index: int, destroy: bool = False) -> float:
        """Crash device ``index`` (deleting its storage first if ``destroy``); wait for detection.

        Returns:
            Seconds from the command to the controller taking the device for dead.
        """
        with self._lock:
            self.devices[index].killed_at = time.monotonic()
        self._command(index, "die", {"wipe": destroy})
        self.log("destroyed" if destroy else "crashed", device=index)
        return self.await_death(index)

    def await_death(self, index: int, timeout: float = WAIT) -> float:
        """Wait until device ``index`` is taken for dead; seconds since it was told to die."""
        deadline = time.monotonic() + timeout
        while self.devices[index].alive:
            if time.monotonic() > deadline:
                raise TimeoutError(f"device {index} is still alive")
            time.sleep(0.05)
        killed = self.devices[index].killed_at
        return float("nan") if killed is None else time.monotonic() - killed

    def regrow(self, lost: Sequence[int]) -> dict[str, Any]:
        """Regrow the children of ``lost`` on the new devices registered under their indices.

        Raises:
            ShardError: If no regrowth data the set holds can rebuild this loss.
        """
        lost = sorted(lost)
        kind = self._regrowth_kind(lost)
        survivors = [k for k in self.live() if k not in lost]
        common = [MANIFEST_NAME, *(regrowth_manifest_name(k) for k in self.sidecars)]
        common += [SCALES_NAME] if self.manifest.scales_sha256 else []
        files = {k: [shard_name(k), regrowth_name(kind, k)] for k in survivors}
        files[survivors[0]] += common
        info = {
            "lost": lost,
            "kind": str(kind),
            "restore": [str(k) for k in self.sidecars if k is not kind],
            "files": {str(k): names for k, names in files.items()},
            "sources": {str(k): list(self.devices[k].address) for k in survivors},
            "common": common,
        }
        self.log("regrowing", lost=lost, kind=str(kind), survivors=survivors)
        requests = {k: self._command(k, "regrow", info) for k in lost}
        report = {"kind": str(kind), "survivors": survivors, "devices": {}}
        for k, request in requests.items():
            reply = self._reply(request, k, ("regrown",))
            fields = reply.info if reply is not None else {"sha256": None}
            fields["matches"] = fields["sha256"] == self.manifest.shard_sha256[k]
            self.log("regrown", device=k, **fields)
            report["devices"][k] = fields
        report["admitted"] = self.admit(lost)
        return report

    def stop(self) -> None:
        """Stop every live device and the controller."""
        with self._lock:
            alive = [k for k, d in self.devices.items() if d.alive]
        for k in alive:
            self._command(k, "stop", {})
        self._stopped.set()
        self.log("stopped", devices=alive)
        self._log_file.close()

    def log(self, event: str, **fields: Any) -> None:
        """Append one event to the log, with the wall-clock time."""
        line = json.dumps({"time": time.time(), "event": event, **fields}, default=str)
        with self._lock:
            if not self._log_file.closed:
                self._log_file.write(line + "\n")
                self._log_file.flush()

    def _route(
        self, ids: list[int], live: list[int], force: dict[str, int] | None
    ) -> dict[str, Any] | None:
        """One attempt at a request; ``None`` if a member it needed died first."""
        if force is None and len(live) == 1:
            return self._answer("solo", ids, live, "alone")
        if force is None:
            pair = live[:2]
            request = self._send_all(pair, "solo", self._request_info(ids, pair))
            answers = self._collect(request, pair, "answer")
            if len(answers) < len(pair):
                return None
            first, second = (answers[k].info for k in pair)
            if gsm8k.is_correct(gsm8k.extract(first["text"]), gsm8k.extract(second["text"]) or ""):
                return _record("agreed", pair, first)
        return self._answer("fuse", ids, live, "fused", force)

    def _answer(
        self,
        kind: str,
        ids: list[int],
        members: list[int],
        route: str,
        force: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        info = self._request_info(ids, members)
        request = next(self._requests)
        for k in members:
            extra = {"die_after": force["die_after"]} if force and force["victim"] == k else {}
            self._command(k, kind, info | extra, request)
        answers = self._collect(request, members, "answer")
        if not answers:
            return None
        return _record(route, members, answers[min(answers)].info)

    def _request_info(self, ids: list[int], members: list[int]) -> dict[str, Any]:
        addresses = {str(k): list(self.devices[k].address) for k in members}
        return {
            "prompt": ids,
            "max_new_tokens": self.max_new_tokens,
            "members": members,
            "addresses": addresses,
        }

    def _regrowth_kind(self, lost: list[int]) -> Redundancy:
        """The first regrowth data that can rebuild ``lost``; each refusal is logged."""
        size = self.manifest.scheme.num_children
        for kind in sorted(self.sidecars, key=list(Redundancy).index):
            try:
                check_losses(lost, size, kind, self.sidecars[kind].tolerated)
            except ShardError as error:
                self.log("regrowth-refused", lost=lost, kind=str(kind), reason=str(error))
                continue
            return kind
        raise ShardError(f"no regrowth data held can rebuild children {lost}")

    def _send_all(self, members: list[int], kind: str, info: dict[str, Any]) -> int:
        request = next(self._requests)
        for k in members:
            self._command(k, kind, info, request)
        return request

    def _command(
        self, index: int, kind: str, info: dict[str, Any], request: int | None = None
    ) -> int:
        request = next(self._requests) if request is None else request
        message = Message(kind, CONTROLLER, info=info | {"request": request})
        self.endpoint.send(index, self.devices[index].address, message)
        return request

    def _reply(self, request: int, index: int, kinds: Sequence[str]) -> Message | None:
        """Device ``index``'s reply to ``request``, or ``None`` if it dies first."""
        deadline = time.monotonic() + WAIT
        while self.devices[index].alive and time.monotonic() < deadline:
            message = self.endpoint.inbox.take(
                lambda m: m.sender == index and m.kind in kinds and m.info["request"] == request,
                0.2,
            )
            if message is not None:
                return message
        return None

    def _collect(self, request: int, members: list[int], kind: str) -> dict[int, Message]:
        """Replies of ``kind`` to ``request`` from every member that stays alive."""
        replies: dict[int, Message] = {}
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            waiting = [k for k in members if k not in replies and self.devices[k].alive]
            if not waiting:
                break
            message = self.endpoint.inbox.take(
                lambda m, w=tuple(waiting): (
                    m.kind == kind and m.info.get("request") == request and m.sender in w
                ),
                0.2,
            )
            if message is not None:
                replies[message.sender] = message
        return replies

    def _registered(self, index: int) -> bool:
        with self._lock:
            return index in self.devices and self.devices[index].alive

    def _monitor(self) -> None:
        """Record registrations and heartbeats; take silent devices for dead."""
        while not self._stopped.is_set():
            message = self.endpoint.inbox.take(lambda m: m.kind in ("hello", "beat"), 0.1)
            if message is not None:
                self._heard(message)
            self._check_silence()

    def _heard(self, message: Message) -> None:
        now, k = time.monotonic(), message.sender
        if message.kind == "hello":
            with self._lock:
                self.devices[k] = DeviceState(tuple(message.info["address"]), now)
            self.log("registered", device=k, has_shard=message.info["has_shard"])
            return
        with self._lock:
            device = self.devices.get(k)
            if device is not None and device.alive:
                device.last_beat = now
                device.peak = max(device.peak, message.info["peak"])

    def _check_silence(self) -> None:
        now, dead = time.monotonic(), []
        with self._lock:
            for k, device in self.devices.items():
                if device.alive and now - device.last_beat > DEAD_AFTER:
                    device.alive = device.ready = False
                    dead.append((k, now - device.last_beat))
        for k, silence in dead:
            self.log("dead", device=k, silence=silence)


def _record(route: str, members: list[int], answer: dict[str, Any]) -> dict[str, Any]:
    keep = ("request", "text", "tokens", "peak", "events", "timing")
    record = {"route": route, "members": members} | {k: answer[k] for k in keep if k in answer}
    record["finished_with"] = answer.get("members", members)
    return record
