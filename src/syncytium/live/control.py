"""What every controller of live devices does (``docs/preregistration-demo.md``).

It tracks which devices are alive from their heartbeats, admits a device's model only if its
file matches the hash the controller expects, kills and destroys devices on the scenario's
schedule, and writes every event to a JSON-lines log, from which the timeline figure and the
H18 checks are computed. The swarm (:mod:`syncytium.live.controller`) and the pipeline
baseline (:mod:`syncytium.live.pipeline`) add how requests are served and losses healed.
The controller itself is never failed.
"""

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

DEAD_AFTER = 4 * HEARTBEAT  # silence after which a device is taken for dead
WAIT = 3600.0  # longest wait for any device's reply, in seconds


@dataclass
class DeviceState:
    """What the controller knows of one device.

    Attributes:
        address: Where it listens.
        last_beat: When its last heartbeat arrived (monotonic clock).
        alive: Whether it is taken for alive.
        ready: Whether its model is loaded and admitted.
        peak: Largest GPU memory it has reported, in bytes.
        killed_at: When it was told to die, if it was.
        silence: How long it had been silent when taken for dead.
        detected_at: When it was taken for dead.
    """

    address: tuple[str, int]
    last_beat: float
    alive: bool = True
    ready: bool = False
    peak: int = 0
    killed_at: float | None = field(default=None, repr=False)
    silence: float = 0.0
    detected_at: float = 0.0


class Control:
    """Registry, heartbeats, admission, kills and the event log.

    Args:
        endpoint: The controller's endpoint (index :data:`~syncytium.live.device.CONTROLLER`).
        log: The JSON-lines event log to append to.
    """

    def __init__(self, endpoint: Endpoint, log: Path) -> None:
        self.endpoint = endpoint
        self.devices: dict[int, DeviceState] = {}
        self._requests = itertools.count()
        self._log_file = log.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        threading.Thread(target=self._monitor, daemon=True).start()

    def expected_sha256(self, index: int) -> str:
        """SHA-256 the model file of device ``index`` must have to be admitted."""
        raise NotImplementedError

    def live(self) -> list[int]:
        """Devices alive with an admitted model, lowest first."""
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
        """Ask each device to load its model; admit it only if its file has the right hash."""
        requests = {
            k: self._command(k, "load", {"sha256": self.expected_sha256(k)}) for k in indices
        }
        admitted = {}
        for k, request in requests.items():
            reply = self._reply(request, k, ("ready", "refused"))
            admitted[k] = reply is not None and reply.kind == "ready"
            with self._lock:
                self.devices[k].ready = admitted[k]
            fields = reply.info if reply is not None else {"sha256": None}
            self.log("admitted" if admitted[k] else "refused", device=k, **fields)
        return admitted

    def kill(self, indices: Sequence[int], destroy: bool = False) -> dict[int, float]:
        """Crash devices together (deleting their storage first if ``destroy``).

        Returns:
            Per device, seconds from the command to the controller taking it for dead.
        """
        for k in indices:
            with self._lock:
                self.devices[k].killed_at = time.monotonic()
            self._command(k, "die", {"wipe": destroy})
            self.log("destroyed" if destroy else "crashed", device=k)
        detected = {k: self.await_death(k) for k in indices}
        for k, seconds in detected.items():
            self.log("detected", device=k, seconds=seconds)
        return detected

    def await_death(self, index: int, timeout: float = WAIT) -> float:
        """Wait until device ``index`` is taken for dead.

        Returns:
            Seconds from being told to die to being taken for dead; for a device that crashed
            on its own, seconds from its last heartbeat.
        """
        deadline = time.monotonic() + timeout
        while self.devices[index].alive:
            if time.monotonic() > deadline:
                raise TimeoutError(f"device {index} is still alive")
            time.sleep(0.05)
        device = self.devices[index]
        if device.killed_at is None:
            return device.silence
        return device.detected_at - device.killed_at

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

    def _send_all(self, members: Sequence[int], kind: str, info: dict[str, Any]) -> int:
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

    def _collect(self, request: int, members: Sequence[int], kind: str) -> dict[int, Message]:
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
            self.log("registered", device=k, has_model=message.info["has_model"])
            return
        with self._lock:
            device = self.devices.get(k)
            if device is None or not device.alive:
                return
            device.last_beat = now
            higher = message.info["peak"] > device.peak
            device.peak = max(device.peak, message.info["peak"])
        if higher:
            self.log("peak", device=k, bytes=message.info["peak"])

    def _check_silence(self) -> None:
        now, dead = time.monotonic(), []
        with self._lock:
            for k, device in self.devices.items():
                if device.alive and now - device.last_beat > DEAD_AFTER:
                    device.alive = device.ready = False
                    device.silence, device.detected_at = now - device.last_beat, now
                    dead.append((k, device.silence))
        for k, silence in dead:
            self.log("dead", device=k, silence=silence)
