"""One node's launcher for live devices (``docs/preregistration-demo.md``).

The scenario runner starts one host on every node of the job (``python -m
syncytium.live.host``). A host registers with the runner and then does only what it is told:
copy files into a device's node-local storage, start a device process on one of its GPUs, or
remove a storage directory. Devices are separate processes, so a device that crashes or is
destroyed takes nothing else with it, and a new device is simply a new process on a free GPU.
"""

import argparse
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from syncytium.live.wire import Endpoint, Message

RUNNER = 98  # the scenario runner's index on the wire
IDLE = 60.0  # seconds between checks that the runner is still there


class Host:
    """Starts and provisions devices on one node.

    Args:
        index: This host's index on the wire.
        runner: The runner's address.
        gpus: The GPU ids this node's devices may use, as CUDA_VISIBLE_DEVICES names them.
    """

    def __init__(self, index: int, runner: tuple[str, int], gpus: Sequence[str]) -> None:
        self.index, self.runner, self.gpus = index, runner, list(gpus)
        self.endpoint = Endpoint(index)
        self.processes: list[subprocess.Popen[bytes]] = []

    def run(self) -> None:
        """Register, then obey the runner until told to stop or the runner is gone.

        A scenario can leave its hosts idle for hours between steps, so silence alone never
        stops a host; ending the host ends the devices it started.
        """
        info = {"address": list(self.endpoint.address), "node": socket.gethostname()}
        self._tell("host-hello", info | {"gpus": self.gpus})
        while True:
            command = self.endpoint.inbox.take(lambda m: m.sender == RUNNER, IDLE)
            if command is None:
                if self._runner_alive():
                    continue
                break
            if command.kind == "stop":
                break
            reply = {"provision": _provision, "spawn": self._spawn, "clear": _clear}[command.kind](
                command.info
            )
            self._tell("done", {"request": command.info["request"]} | reply)
        for process in self.processes:
            if process.poll() is None:
                process.kill()
        self.endpoint.close()

    def _spawn(self, info: dict[str, Any]) -> dict[str, Any]:
        """Start ``info["argv"]`` on GPU ``info["gpu"]``, logging to ``info["log"]``."""
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(info["gpu"])}
        Path(info["log"]).parent.mkdir(parents=True, exist_ok=True)
        with Path(info["log"]).open("ab") as log:
            process = subprocess.Popen(info["argv"], env=env, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(process)
        return {"pid": process.pid, "started": time.time()}

    def _runner_alive(self) -> bool:
        """Whether the runner still accepts connections."""
        try:
            socket.create_connection(self.runner, timeout=5).close()
        except OSError:
            return False
        return True

    def _tell(self, kind: str, info: dict[str, Any]) -> None:
        self.endpoint.send(RUNNER, self.runner, Message(kind, self.index, info=info))


def _provision(info: dict[str, Any]) -> dict[str, Any]:
    """Copy ``info["files"]`` into the directory ``info["storage"]``."""
    storage = Path(info["storage"])
    storage.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in info["files"]:
        shutil.copy(source, storage / Path(source).name)
        copied += Path(source).stat().st_size
    return {"bytes": copied}


def _clear(info: dict[str, Any]) -> dict[str, Any]:
    """Remove the directory ``info["storage"]`` and everything in it."""
    shutil.rmtree(info["storage"], ignore_errors=True)
    return {}


def main(argv: Sequence[str] | None = None) -> None:
    """Run this node's host until the runner stops it."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--runner", required=True, help="HOST:PORT")
    parser.add_argument("--index", type=int, help="default: 100 + the SLURM node id")
    args = parser.parse_args(argv)
    index = args.index if args.index is not None else 100 + int(os.environ.get("SLURM_NODEID", 0))
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    host, port = args.runner.rsplit(":", 1)
    Host(index, (host, int(port)), gpus).run()


if __name__ == "__main__":
    main()
