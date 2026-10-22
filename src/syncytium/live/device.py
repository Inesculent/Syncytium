"""A live device (``docs/preregistration-demo.md``): one child, one GPU, one storage directory.

A device registers with the controller and sends heartbeats. It answers requests alone or as
part of a fused group, and runs probes. It serves its files to devices that need them for
regrowth. When told to, it crashes (optionally destroying its storage first), or regrows its
own child from the survivors' files and verifies it against the manifest's hash.

On the cluster each device is its own process (``python -m syncytium.live.device``); tests run
devices as threads, with a crash hook that stops the thread instead of the process.
"""

import argparse
import hashlib
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PretrainedConfig

from syncytium.live.exchange import DroppedError, Exchange
from syncytium.live.packed import load_member, packed_bytes, set_exchange
from syncytium.live.wire import Endpoint, Link, Message
from syncytium.shards import (
    MANIFEST_NAME,
    Manifest,
    Redundancy,
    regrow_shards,
    restore_regrowth_files,
    shard_name,
)
from syncytium.shards.manifest import file_sha256
from syncytium.shards.regrow import regrowth_name
from syncytium.tasks import gsm8k
from syncytium.tasks.generation import Decoding, generate

CONTROLLER = 99
HEARTBEAT = 0.5  # seconds between heartbeats
CHUNK = 1 << 26  # bytes per file-transfer message
WORK = "regrow-work"  # where survivors' files are copied while regrowing
PROBE_TAIL = 16  # positions whose logits a probe returns


class CrashedError(Exception):
    """Raised by a test's crash hook to stop a device thread as a crash stops a process."""


@dataclass(frozen=True)
class DeviceSettings:
    """Where a device runs and whom it reports to.

    Attributes:
        index: The member it plays (its child's index).
        storage: Its storage directory.
        controller: The controller's address.
        device: Torch device for its model.
        memory_gb: Cap on its GPU memory, or ``None``.
        timeout: Seconds to wait for another device before taking it for dead.
        stage: The pipeline stage it holds, for a device of the baseline; else ``None``.
    """

    index: int
    storage: Path
    controller: tuple[str, int]
    device: str = "cuda"
    memory_gb: float | None = None
    timeout: float = 5.0
    stage: int | None = None


class Device:
    """One live device.

    Args:
        settings: Where it runs and whom it reports to.
        endpoint: Its network endpoint, whose index is ``settings.index``.
        tokenizer: The model's tokenizer; defaults to the manifest model's.
        model_config: The model's config; defaults to the manifest model's (tests pass one).
        crash: Called to crash; defaults to ending the process at once.
    """

    def __init__(
        self,
        settings: DeviceSettings,
        endpoint: Endpoint,
        tokenizer: Any = None,
        model_config: PretrainedConfig | None = None,
        crash: Callable[[], None] | None = None,
    ) -> None:
        self.settings, self.endpoint = settings, endpoint
        self.index = settings.index
        self.tokenizer, self.model_config = tokenizer, model_config
        self.crash = crash or (lambda: os._exit(1))
        self.model: Any = None
        self.address: tuple[str, int] = endpoint.address
        # Heartbeats and commands never queue behind data on an emulated link.
        endpoint.direct = frozenset({CONTROLLER})
        self._stopped = threading.Event()
        if settings.device.startswith("cuda") and settings.memory_gb:
            total = torch.cuda.get_device_properties(settings.device).total_memory
            torch.cuda.set_per_process_memory_fraction(settings.memory_gb * 1e9 / total)

    def run(self) -> None:
        """Register, then serve the controller's commands until stopped or crashed."""
        info = {"address": list(self.address), "has_model": self._model_file().exists()}
        self._tell("hello", info)
        for target in (self._beat, self._serve_files, *self.services()):
            threading.Thread(target=target, daemon=True).start()
        kinds = set(self.handlers())
        try:
            while not self._stopped.is_set():
                command = self.endpoint.inbox.take(
                    lambda m: m.sender == CONTROLLER and m.kind in kinds, 0.5
                )
                if command is not None:
                    self._handle(command)
        except CrashedError:
            pass
        finally:
            self._stopped.set()
            self.endpoint.close()

    def handlers(self) -> dict[str, Callable[[Message], None]]:
        """The controller's commands this device obeys, by kind."""
        return {
            "load": self._load,
            "solo": self._solo,
            "fuse": self._fuse,
            "probe": self._probe,
            "die": self._die,
            "regrow": self._regrow,
            "stop": lambda _: self._stopped.set(),
        }

    def _handle(self, command: Message) -> None:
        self.handlers()[command.kind](command)

    def _load(self, command: Message) -> None:
        """Load its child if its shard matches the hash the controller expects."""
        path = self._model_file()
        actual = file_sha256(path) if path.exists() else None
        if actual != command.info["sha256"]:
            self._tell("refused", {"request": command.info["request"], "sha256": actual})
            return
        self.model = load_member(
            self.settings.storage, self.index, self.settings.device, self.model_config
        )
        self._ensure_tokenizer()
        self._warm_up(lambda ids: self.model(input_ids=ids))
        info = {"request": command.info["request"], "sha256": actual}
        self._tell("ready", info | {"packed_bytes": packed_bytes(self.model)})

    def _warm_up(self, forward: Callable[[torch.Tensor], Any]) -> None:
        """Run one token alone, so start-up costs never land in a timed or fused pass."""
        with torch.no_grad():
            forward(torch.tensor([[2]], device=self.settings.device))

    def _ensure_tokenizer(self) -> None:
        """Load the manifest model's tokenizer unless one was given."""
        if self.tokenizer is None:
            manifest = Manifest.load(self.settings.storage / MANIFEST_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(manifest.model)

    def _solo(self, command: Message) -> None:
        start = time.monotonic()
        completion = self._answer(command.info)
        self._reply_answer(command, completion, start, None)

    def _fuse(self, command: Message) -> None:
        start = time.monotonic()
        exchange = self._exchange(command.info)
        die_after = command.info.get("die_after")

        def observe(step: int) -> None:
            if step == die_after:
                self.crash()

        set_exchange(self.model, exchange.average)
        try:
            completion = self._answer(command.info, observe if die_after is not None else None)
        except DroppedError:  # the leader went on without it; its answer is not the group's
            completion = None
        finally:
            set_exchange(self.model, None)
        self._reply_answer(command, completion, start, exchange)

    def _probe(self, command: Message) -> None:
        """Teacher-force a prompt through the group; report top-1 tokens and the logits' hash."""
        exchange = self._exchange(command.info)
        set_exchange(self.model, exchange.average)
        try:
            ids = torch.tensor([command.info["prompt"]], device=self.settings.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids).logits[0]
        except DroppedError:
            info = {"request": command.info["request"], "dropped": True, "events": exchange.events}
            self._tell("probe", info)
            return
        finally:
            set_exchange(self.model, None)
        raw = logits.detach().contiguous().cpu()
        digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()
        info = {
            "request": command.info["request"],
            "top1": raw.argmax(dim=-1).tolist(),
            "sha256": digest,
            "members": exchange.members,
            "events": exchange.events,
            "timing": summarize_timing(exchange.timing),
        }
        self._tell("probe", info, raw[-PROBE_TAIL:].float())

    def _die(self, command: Message) -> None:
        if command.info.get("wipe"):
            for entry in self.settings.storage.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        self.crash()

    def _regrow(self, command: Message) -> None:
        """Fetch the survivors' files, regrow every lost child, and keep only its own."""
        start = time.monotonic()
        info, storage = command.info, self.settings.storage
        work = storage / WORK
        work.mkdir(parents=True, exist_ok=True)
        moved = self._fetch_all(info, work)
        lost = info["lost"]
        regrow_shards(work, lost, Redundancy(info["kind"]))
        for kind in info["restore"]:
            restore_regrowth_files(work, lost, Redundancy(kind))
        keep = [shard_name(self.index), *info["common"]]
        keep += [regrowth_name(Redundancy(k), self.index) for k in (info["kind"], *info["restore"])]
        for name in keep:
            shutil.move(work / name, storage / name)
        shutil.rmtree(work)
        self._tell(
            "regrown",
            {
                "request": info["request"],
                "sha256": file_sha256(self._model_file()),
                "bytes": moved,
                "seconds": time.monotonic() - start,
            },
        )

    def services(self) -> list[Callable[[], None]]:
        """Loops run in their own threads besides heartbeats and file serving."""
        return []

    def _fetch_all(self, info: dict[str, Any], into: Path) -> int:
        """Copy ``info["files"]`` (names by source) from ``info["sources"]``; bytes moved."""
        moved = 0
        for source, names in info["files"].items():
            for name in names:
                moved += self._fetch(int(source), tuple(info["sources"][source]), name, into)
        return moved

    def _fetch(self, source: int, address: tuple[str, int], name: str, work: Path) -> int:
        """Copy file ``name`` from device ``source`` into ``work``; return its size."""
        request = Message("fetch", self.index, info={"name": name, "address": list(self.address)})
        self.endpoint.send(source, address, request)
        size = 0
        with (work / name).open("wb") as file:
            while True:
                chunk = self.endpoint.inbox.take(
                    lambda m: m.kind == "chunk" and m.sender == source and m.info["name"] == name,
                    120,
                )
                if chunk is None:
                    raise RuntimeError(f"device {source} stopped sending {name}")
                data = chunk.tensor.numpy().tobytes()
                file.write(data)
                size += len(data)
                if chunk.info["last"]:
                    return size

    def _serve_files(self) -> None:
        """Send any file of its storage to a device that asks for it, in chunks."""
        while not self._stopped.is_set():
            request = self.endpoint.inbox.take(lambda m: m.kind == "fetch", 0.5)
            if request is None:
                continue
            name, address = request.info["name"], tuple(request.info["address"])
            data = (self.settings.storage / name).read_bytes()
            for offset in range(0, max(len(data), 1), CHUNK):
                piece = bytearray(data[offset : offset + CHUNK])
                info = {"name": name, "last": offset + CHUNK >= len(data)}
                chunk = Message(
                    "chunk", self.index, -1, info, torch.frombuffer(piece, dtype=torch.uint8)
                )
                self.endpoint.send(request.sender, address, chunk)

    def _beat(self) -> None:
        while not self._stopped.is_set():
            self._tell("beat", {"peak": self._peak()})
            self._stopped.wait(HEARTBEAT)

    def _answer(self, info: dict[str, Any], observe: Callable[[int], None] | None = None) -> Any:
        decoding = Decoding(info["max_new_tokens"], gsm8k.STOPS)
        return generate(self.model, self.tokenizer, [info["prompt"]], decoding, observe)[0]

    def _exchange(self, info: dict[str, Any]) -> Exchange:
        addresses = {int(m): tuple(a) for m, a in info["addresses"].items()}
        members, timeout = info["members"], self.settings.timeout
        return Exchange(self.endpoint, addresses, members, timeout, info["request"])

    def _reply_answer(
        self, command: Message, completion: Any, start: float, exchange: Exchange | None
    ) -> None:
        info = {
            "request": command.info["request"],
            "text": None if completion is None else completion.text,
            "tokens": 0 if completion is None else len(completion.tokens),
            "seconds": time.monotonic() - start,
            "peak": self._peak(),
            "dropped": completion is None,
        }
        if exchange is not None:
            info |= {"members": exchange.members, "events": exchange.events}
            info["timing"] = summarize_timing(exchange.timing)
        self._tell("answer", info)

    def _tell(self, kind: str, info: dict[str, Any], tensor: torch.Tensor | None = None) -> None:
        message = Message(kind, self.index, info=info, tensor=tensor)
        self.endpoint.send(CONTROLLER, self.settings.controller, message)

    def _model_file(self) -> Path:
        return self.settings.storage / shard_name(self.index)

    def _peak(self) -> int:
        if self.settings.device.startswith("cuda"):
            return torch.cuda.max_memory_reserved(self.settings.device)
        return 0


def summarize_timing(timing: Sequence[tuple[int, int, float]]) -> list[list[float]]:
    """Averaging rounds grouped by (values, live members): ``[values, live, rounds, seconds]``.

    The first round of a request is left out, since it also waits for every member to start.
    """
    groups: dict[tuple[int, int], list[float]] = {}
    for values, live, seconds in timing[1:]:
        group = groups.setdefault((values, live), [0, 0.0])
        group[0] += 1
        group[1] += seconds
    return [[values, live, n, total] for (values, live), (n, total) in sorted(groups.items())]


def device_from_args(argv: Sequence[str] | None = None, **overrides: Any) -> Device:
    """The device a command line describes; ``overrides`` go to its constructor."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--controller", required=True, help="HOST:PORT")
    parser.add_argument("--host", default="", help="interface to listen on; default all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--stage", type=int, help="the pipeline stage it holds (baseline)")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for a peer")
    parser.add_argument("--latency", type=float, default=0.0, help="emulated, seconds")
    parser.add_argument("--bandwidth", type=float, default=0.0, help="emulated, bits per second")
    args = parser.parse_args(argv)
    host, port = args.controller.rsplit(":", 1)
    settings = DeviceSettings(
        args.index,
        args.storage,
        (host, int(port)),
        args.device,
        args.memory_gb,
        args.timeout,
        args.stage,
    )
    endpoint = Endpoint(args.index, link=Link(args.latency, args.bandwidth), host=args.host)
    args.storage.mkdir(parents=True, exist_ok=True)
    if args.stage is None:
        return Device(settings, endpoint, **overrides)
    from syncytium.live.pipeline import StageDevice  # it imports this module

    return StageDevice(settings, endpoint, **overrides)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one device until the controller stops it."""
    device_from_args(argv).run()


if __name__ == "__main__":
    main()
