"""Messages between live devices over TCP (``docs/preregistration-demo.md``).

A message is a small JSON header and an optional tensor, framed by their lengths. Every device
listens on one port; reader threads put incoming messages in an :class:`Inbox`, where the
computing thread waits for the one it needs. Sends can be slowed to emulate a network link
(:class:`Link`), so a run on the cluster can stand in for a wired LAN.
"""

import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

_HEADER = struct.Struct("!IQ")  # header length, payload length


@dataclass
class Message:
    """One message between devices.

    Attributes:
        kind: What it is, such as "contrib" or "result".
        sender: The sending device's member index.
        round: The averaging round it belongs to, or -1.
        info: Any other JSON-serializable fields.
        tensor: An optional tensor payload.
        deliver_at: When the receiver may use it (monotonic clock); set on receipt.
        size: Its framed size in bytes; set on receipt.
    """

    kind: str
    sender: int
    round: int = -1
    info: dict[str, Any] = field(default_factory=dict)
    tensor: Tensor | None = None
    deliver_at: float = 0.0
    size: int = 0


@dataclass(frozen=True)
class Link:
    """An emulated full-duplex link from a device to a switch.

    Each end serializes its own traffic at ``bandwidth``: a device sends one message after
    another, and messages arriving together are received one after another. Every message
    then takes ``latency`` in flight, in parallel with others. So a leader sending *k* vectors
    of *S* bits finishes after *k S*/``bandwidth`` plus one ``latency``, as on a switched LAN.

    Attributes:
        latency: Seconds every message spends in flight.
        bandwidth: Bits per second at each end; 0 means unlimited.
    """

    latency: float = 0.0
    bandwidth: float = 0.0

    def serialize(self, size: int) -> float:
        """Seconds to put a message of ``size`` bytes on, or take it off, the link."""
        return size * 8 / self.bandwidth if self.bandwidth else 0.0


def encode(message: Message) -> bytes:
    """The framed bytes of ``message``."""
    header = {"kind": message.kind, "sender": message.sender, "round": message.round}
    header["info"] = message.info
    payload = b""
    if message.tensor is not None:
        tensor = message.tensor.detach().contiguous().cpu()
        header["dtype"] = str(tensor.dtype).removeprefix("torch.")
        header["shape"] = list(tensor.shape)
        payload = tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
    raw = json.dumps(header).encode()
    return _HEADER.pack(len(raw), len(payload)) + raw + payload


def decode(header_bytes: bytes, payload: bytes) -> Message:
    """The message framed as ``header_bytes`` and ``payload``."""
    header = json.loads(header_bytes)
    tensor = None
    if "dtype" in header:
        dtype = getattr(torch, header["dtype"])
        flat = torch.frombuffer(bytearray(payload), dtype=torch.uint8) if payload else None
        tensor = (
            flat.view(dtype).reshape(header["shape"])
            if flat is not None
            else torch.empty(header["shape"], dtype=dtype)
        )
    return Message(header["kind"], header["sender"], header["round"], header["info"], tensor)


def read_message(sock: socket.socket) -> Message | None:
    """The next message on ``sock``, or ``None`` once it is closed."""
    head = _read_exactly(sock, _HEADER.size)
    if head is None:
        return None
    header_length, payload_length = _HEADER.unpack(head)
    header = _read_exactly(sock, header_length)
    payload = _read_exactly(sock, payload_length) if payload_length else b""
    if header is None or payload is None:
        return None
    message = decode(header, payload)
    message.size = _HEADER.size + header_length + payload_length
    return message


class Inbox:
    """Messages received but not yet taken, which threads wait on by predicate."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._changed = threading.Condition()

    def put(self, message: Message) -> None:
        """Add a received message and wake every waiting thread."""
        with self._changed:
            self._messages.append(message)
            self._changed.notify_all()

    def take(self, match: Callable[[Message], bool], timeout: float) -> Message | None:
        """Remove and return the first delivered message ``match`` accepts; ``None`` on timeout.

        A message is delivered once its :attr:`Message.deliver_at` has passed.
        """
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                now = time.monotonic()
                due = [m for m in self._messages if match(m)]
                for message in due:
                    if message.deliver_at <= now:
                        self._messages.remove(message)
                        return message
                wake = min([deadline, *(m.deliver_at for m in due)])
                if now >= deadline:
                    return None
                self._changed.wait(wake - now)


class Endpoint:
    """A device's network presence: a listening socket, an inbox, and connections to peers.

    Args:
        index: This device's member index.
        port: Port to listen on; 0 picks a free one (see :attr:`address`).
        link: Emulated link between this device and the switch, applied to everything it
            sends and receives.
        host: Interface to listen on.
    """

    def __init__(self, index: int, port: int = 0, link: Link | None = None, host: str = "") -> None:
        self.index, self.link = index, link or Link()
        self.inbox = Inbox()
        self._server = socket.create_server((host, port))
        # Keyed by address too: a device that restarts listens somewhere new.
        self._peers: dict[tuple[int, tuple[str, int]], socket.socket] = {}
        self._lock = threading.Lock()
        self._link_lock = threading.Lock()
        self._sending_until = self._receiving_until = 0.0
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def address(self) -> tuple[str, int]:
        """The address peers connect to: the host name when listening on every interface."""
        host, port = self._server.getsockname()[:2]
        return (socket.gethostname() if host in ("0.0.0.0", "::") else host), port

    def send(self, peer: int, address: tuple[str, int], message: Message) -> bool:
        """Send ``message`` to ``peer``; ``False`` if it cannot be reached."""
        data = encode(message)
        with self._link_lock:
            start = max(time.monotonic(), self._sending_until)
            self._sending_until = start + self.link.serialize(len(data))
            done = self._sending_until
        time.sleep(max(done - time.monotonic(), 0.0))
        key = (peer, tuple(address))
        try:
            with self._lock:
                if key not in self._peers:
                    self._peers[key] = socket.create_connection(address, timeout=5)
                    self._peers[key].settimeout(None)
                self._peers[key].sendall(data)
        except OSError:
            self._forget(key)
            return False
        return True

    def close(self) -> None:
        """Stop listening and close every connection, as a crashed device would."""
        self._closed = True
        self._server.close()
        with self._lock:
            for sock in self._peers.values():
                sock.close()
            self._peers.clear()

    def _forget(self, key: tuple[int, tuple[str, int]]) -> None:
        with self._lock:
            sock = self._peers.pop(key, None)
        if sock is not None:
            sock.close()

    def _accept(self) -> None:
        while not self._closed:
            try:
                sock, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._receive, args=(sock,), daemon=True).start()

    def _receive(self, sock: socket.socket) -> None:
        with sock:
            while not self._closed:
                try:
                    message = read_message(sock)
                except OSError:
                    return
                if message is None:
                    return
                with self._link_lock:
                    taken = self.link.serialize(message.size)
                    self._receiving_until = max(time.monotonic(), self._receiving_until + taken)
                    message.deliver_at = self._receiving_until + self.link.latency
                self.inbox.put(message)


def _read_exactly(sock: socket.socket, size: int) -> bytes | None:
    chunks, left = [], size
    while left:
        chunk = sock.recv(min(left, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)
