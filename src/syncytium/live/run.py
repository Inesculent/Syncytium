"""Running H18's scenarios on the cluster (``docs/preregistration-demo.md``).

``python -m syncytium.live.run prepare CONFIG`` writes the children with their regrowth data
and the baseline's 6-bit stages. ``... reference CONFIG`` computes the simulation's probe
logits for check (a). ``... run CONFIG --scenario A --network lan`` starts one host per node
of the job, then plays the scenario's schedule against the swarm and then against the
baseline, on the same GPUs, with the same requests. ``... check CONFIG`` computes the checks
from everything run so far (:mod:`syncytium.live.checks`).

Device k runs on node k // 2 and that node's GPU k % 2, so half the links cross the cluster
network. Storage is node-local. A new device starts on the GPU its destroyed predecessor left
free, with an empty directory.
"""

import argparse
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from syncytium.live.control import Control
from syncytium.live.controller import Controller
from syncytium.live.device import CONTROLLER
from syncytium.live.host import RUNNER
from syncytium.live.pipeline import STAGES_NAME, PipelineController, stage_name, write_stages
from syncytium.live.wire import Endpoint, Message
from syncytium.quant import Scheme
from syncytium.shards import (
    MANIFEST_NAME,
    SCALES_NAME,
    Manifest,
    Redundancy,
    ShardError,
    shard_name,
    verify_shard,
    write_regrowth,
)
from syncytium.shards.calibration import shard_model
from syncytium.shards.regrow import regrowth_manifest_name, regrowth_name
from syncytium.sweep.records import append_run_metadata
from syncytium.tasks import gsm8k

DEVICES = 4
ROLES = {0: 0, 1: 1, 2: 0, 3: 1}  # the baseline's stage on each device
SYSTEMS = ("swarm", "pipeline")
NETWORKS = ("cluster", "lan")
VICTIM_WAIT = 60.0  # seconds to wait for E1's victim to be found dead after its request

# Each step: (label, action, devices). The labels are the pre-registration's.
SCENARIOS: dict[str, tuple[tuple[str, str, tuple[int, ...]], ...]] = {
    "A": (
        ("P0", "probe", ()),
        ("P0", "phase", ()),
        ("E1", "crash_during", (2,)),
        ("P1", "phase", ()),
        ("E2", "destroy", (0,)),
        ("P2", "phase", ()),
        ("E3", "rejoin", (2,)),
        ("E4", "new", (0,)),
        ("P3", "probe", ()),
        ("P3", "phase", ()),
    ),
    "B": (
        ("P0", "probe", ()),
        ("P0", "phase", ()),
        ("E1", "crash_during", (2,)),
        ("P1", "phase", ()),
        ("E2", "destroy", (0,)),
        ("P2", "phase", ()),
        ("E3", "wipe", (2,)),
        ("E4", "new", (0, 2)),
        ("P3", "probe", ()),
        ("P3", "phase", ()),
    ),
    "C": (
        ("P0", "probe", ()),
        ("P0", "phase", ()),
        ("E1", "destroy", (0, 1)),
        ("P1", "phase", ()),
        ("E2", "new", (0, 1)),
        ("P2", "probe", ()),
        ("P2", "phase", ()),
    ),
}


@dataclass(frozen=True)
class LiveConfig:
    """A live run's settings (the ``[live]`` table of its config).

    Attributes:
        name: The run's name.
        output: Where results go.
        model: The model's Hugging Face id.
        children: The children's shard set, with both kinds of regrowth data.
        baseline: The 6-bit set, with its stage files.
        storage: Node-local directory under which devices keep their files.
        bits: Bits per child code.
        group_size: Weights per quantization group.
        rotation_seed: Seed of the shared rotation.
        baseline_bits: Bits of the baseline's round-to-nearest model.
        residue_losses: Losses the coded residue survives, t.
        memory_gb: Cap on each device's GPU memory.
        timeout: Seconds a device waits for another before taking it for dead.
        max_new_tokens: Most answer tokens per request.
        per_phase: Requests per phase.
        probe_problems: GSM8K test problems whose prompts make the probe.
        die_after: Answer tokens after which the victim of E1 crashes.
        lan: Emulated link (latency in seconds, bandwidth in bits per second).
        device: Torch device of every live device.
        listen: Interface the controller and devices listen on; empty for every one.
    """

    name: str
    output: Path
    model: str
    children: Path
    baseline: Path
    storage: Path
    bits: int = 3
    group_size: int = 64
    rotation_seed: int = 0
    baseline_bits: int = 6
    residue_losses: int = 2
    memory_gb: float = 8.0
    timeout: float = 5.0
    max_new_tokens: int = 256
    per_phase: int = 8
    probe_problems: int = 16
    die_after: int = 32
    lan: tuple[float, float] = (0.0002, 1e9)
    device: str = "cuda"
    listen: str = ""

    @classmethod
    def load(cls, path: Path) -> "LiveConfig":
        """Read a config file."""
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        live = dict(raw["live"])
        for key in ("children", "baseline", "storage"):
            live[key] = Path(live[key])
        if "lan" in live:
            live["lan"] = (live["lan"]["latency"], live["lan"]["bandwidth"])
        return cls(raw["name"], Path(raw["output"]), raw["model"]["path"], **live)

    def link(self, network: str) -> tuple[float, float]:
        """(latency, bandwidth) added to every device's link on ``network``."""
        return self.lan if network == "lan" else (0.0, 0.0)


def prepare(config: LiveConfig, device: str = "cuda") -> None:
    """Write the children with regrowth data, and the baseline's stages.

    Every step whose output is already complete is skipped, so an interrupted preparation
    resumes where it stopped.
    """
    if not _written(config.children, DEVICES):
        scheme = Scheme(config.bits, config.group_size, DEVICES, 0.5, config.rotation_seed, True)
        shard_model(config.model, scheme, config.children, device)
    if not (config.children / regrowth_manifest_name(Redundancy.NEIGHBOUR)).exists():
        write_regrowth(config.children, Redundancy.NEIGHBOUR)
    if not (config.children / regrowth_manifest_name(Redundancy.RESIDUE)).exists():
        write_regrowth(config.children, Redundancy.RESIDUE, config.residue_losses)
    if not _written(config.baseline, 1):
        rtn = Scheme(config.baseline_bits, config.group_size, 1, 0.5, config.rotation_seed, True)
        shard_model(config.model, rtn, config.baseline, device)
    if not (config.baseline / STAGES_NAME).exists():
        write_stages(config.baseline, 2)


def _written(directory: Path, children: int) -> bool:
    """Whether a shard set with every child matching its manifest is in ``directory``."""
    if not (directory / MANIFEST_NAME).exists():
        return False
    manifest = Manifest.load(directory / MANIFEST_NAME)
    try:
        for k in range(children):
            verify_shard(directory / shard_name(k), manifest, k)
    except (ShardError, FileNotFoundError):
        return False
    return True


def problems(config: LiveConfig) -> tuple[list[gsm8k.Problem], list[gsm8k.Problem]]:
    """The probe's problems and the request stream, in dataset order."""
    every = gsm8k.load_gsm8k("test")
    return every[: config.probe_problems], every[config.probe_problems :]


class Hosts:
    """The hosts of the job's nodes, as seen from the runner.

    Args:
        endpoint: The runner's endpoint.
        count: How many hosts to wait for.
    """

    def __init__(self, endpoint: Endpoint, count: int, timeout: float = 600.0) -> None:
        self.endpoint = endpoint
        self.hosts: dict[int, dict[str, Any]] = {}
        self._requests = 0
        deadline = time.monotonic() + timeout
        while len(self.hosts) < count:
            hello = endpoint.inbox.take(lambda m: m.kind == "host-hello", 5)
            if hello is not None:
                self.hosts[hello.sender] = hello.info
            elif time.monotonic() > deadline:
                raise TimeoutError(f"{len(self.hosts)} of {count} hosts registered")
        self.order = sorted(self.hosts)

    def call(self, node: int, kind: str, info: dict[str, Any]) -> dict[str, Any]:
        """Have host ``node`` (0-based) do ``kind`` and return its reply."""
        index = self.order[node]
        self._requests += 1
        request = self._requests
        address = tuple(self.hosts[index]["address"])
        self.endpoint.send(index, address, Message(kind, RUNNER, info=info | {"request": request}))
        reply = self.endpoint.inbox.take(
            lambda m: m.kind == "done" and m.info["request"] == request, 3600
        )
        if reply is None:
            raise TimeoutError(f"host {index} did not finish {kind}")
        return reply.info

    def gpu(self, node: int, slot: int) -> str:
        """The name of GPU ``slot`` on host ``node``."""
        return self.hosts[self.order[node]]["gpus"][slot]

    def stop(self) -> None:
        """Stop every host."""
        for index in self.order:
            address = tuple(self.hosts[index]["address"])
            self.endpoint.send(index, address, Message("stop", RUNNER))


class Scenario:
    """One scenario on one network against one system.

    Args:
        config: The run's settings.
        hosts: The job's hosts.
        system: "swarm" or "pipeline".
        where: (scenario name, network name).
        out: Where this system's results go.
    """

    def __init__(
        self,
        config: LiveConfig,
        hosts: Hosts,
        system: str,
        where: tuple[str, str],
        out: Path,
    ) -> None:
        self.config, self.hosts, self.system = config, hosts, system
        self.scenario, self.network = where
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.endpoint = Endpoint(CONTROLLER, host=config.listen)
        self.controller = self._controller()
        self.probe_problems, self.stream = problems(config)
        self.next_problem = 0

    def play(self) -> None:
        """Run the schedule, then stop every device and clear their storage."""
        self.controller.log(
            "scenario", scenario=self.scenario, network=self.network, system=self.system
        )
        try:
            self._start_all()
            for label, action, devices in SCENARIOS[self.scenario]:
                self.controller.log("step", label=label, action=action, devices=list(devices))
                self._actions()[action](label, devices)
        finally:
            self.controller.stop()
            self.endpoint.close()
            for k in range(DEVICES):
                self.hosts.call(k // 2, "clear", {"storage": str(self._storage(k))})

    def _actions(self) -> dict[str, Callable[[str, tuple[int, ...]], None]]:
        return {
            "probe": lambda label, _: self._probe(label),
            "phase": lambda label, _: self._phase(label),
            "crash_during": self._crash_during,
            "destroy": lambda _, devices: self.controller.kill(devices, destroy=True),
            "wipe": self._wipe,
            "rejoin": self._rejoin,
            "new": self._new,
        }

    def _controller(self) -> Control:
        tokenizer = _tokenizer(self.config.model)
        log = self.out / "events.jsonl"
        if self.system == "swarm":
            return Controller(
                self.endpoint, self.config.children, tokenizer, log, self.config.max_new_tokens
            )
        return PipelineController(
            self.endpoint, self.config.baseline, tokenizer, log, ROLES, self.config.max_new_tokens
        )

    def _start_all(self) -> None:
        for k in range(DEVICES):
            files = [str(self._source() / name) for name in self._files(k)]
            self.hosts.call(k // 2, "provision", {"storage": str(self._storage(k)), "files": files})
            self._spawn(k)
        self.controller.await_devices(range(DEVICES))
        self.controller.admit(range(DEVICES))

    def _source(self) -> Path:
        return self.config.children if self.system == "swarm" else self.config.baseline

    def _files(self, k: int) -> list[str]:
        if self.system == "pipeline":
            return [MANIFEST_NAME, SCALES_NAME, STAGES_NAME, stage_name(ROLES[k])]
        sidecars = [regrowth_manifest_name(r) for r in Redundancy]
        own = [shard_name(k), *(regrowth_name(r, k) for r in Redundancy)]
        return [MANIFEST_NAME, SCALES_NAME, *sidecars, *own]

    def _storage(self, k: int) -> Path:
        where = f"{self.config.name}-{self.scenario}-{self.network}-{self.system}"
        return self.config.storage / where / f"device-{k}"

    def _spawn(self, k: int) -> None:
        latency, bandwidth = self.config.link(self.network)
        host, port = self.endpoint.address
        argv = [sys.executable, "-m", "syncytium.live.device", "--index", str(k)]
        argv += ["--storage", str(self._storage(k)), "--controller", f"{host}:{port}"]
        argv += ["--memory-gb", str(self.config.memory_gb), "--timeout", str(self.config.timeout)]
        argv += ["--latency", str(latency), "--bandwidth", str(bandwidth)]
        argv += ["--device", self.config.device, "--host", self.config.listen]
        argv += ["--stage", str(ROLES[k])] if self.system == "pipeline" else []
        info = {"argv": argv, "gpu": self.hosts.gpu(k // 2, k % 2)}
        info["log"] = str(self.out / "devices" / f"device-{k}.log")
        reply = self.hosts.call(k // 2, "spawn", info)
        self.controller.log("spawned", device=k, pid=reply["pid"], gpu=info["gpu"])

    def _request(self, label: str, force: dict[str, int] | None = None) -> None:
        problem = self.stream[self.next_problem]
        tags = {"phase": label, "problem": self.config.probe_problems + self.next_problem}
        self.next_problem += 1
        record = self.controller.serve(gsm8k.prompt(problem.question), force, tags)
        correct = gsm8k.is_correct(record["prediction"], problem.answer)
        self.controller.log("graded", request=record.get("request"), correct=correct, **tags)

    def _phase(self, label: str) -> None:
        for _ in range(self.config.per_phase):
            self._request(label)

    def _probe(self, label: str) -> None:
        if self.system != "swarm":
            return
        prompts = [gsm8k.prompt(p.question) for p in self.probe_problems]
        result = self.controller.probe(prompts)
        if label != "P0":
            for prompt in result["prompts"]:
                prompt.pop("tail")
        torch.save(result, self.out / f"probe-{label}.pt")

    def _crash_during(self, label: str, devices: tuple[int, ...]) -> None:
        (victim,) = devices
        self._request(label, {"victim": victim, "die_after": self.config.die_after})
        try:
            self._detect({victim: self.controller.await_death(victim, VICTIM_WAIT)})
        except TimeoutError:
            # Its answer ended before it was due to crash: crash it now, and say so.
            self.controller.log("victim-survived", device=victim)
            self.controller.kill([victim])

    def _detect(self, detected: dict[int, float]) -> None:
        for k, seconds in detected.items():
            self.controller.log("detected", device=k, seconds=seconds)

    def _wipe(self, _: str, devices: tuple[int, ...]) -> None:
        """Destroy the storage of devices that are already down."""
        for k in devices:
            self.hosts.call(k // 2, "clear", {"storage": str(self._storage(k))})
            self.controller.log("destroyed", device=k, already_down=True)

    def _rejoin(self, _: str, devices: tuple[int, ...]) -> None:
        for k in devices:
            self._spawn(k)
        self.controller.await_devices(devices)
        self.controller.admit(devices)

    def _new(self, _: str, devices: tuple[int, ...]) -> None:
        for k in devices:
            self.hosts.call(k // 2, "clear", {"storage": str(self._storage(k))})
            self._spawn(k)
        self.controller.await_devices(devices)
        try:
            report = self.controller.heal(devices)
        except (ShardError, TimeoutError, RuntimeError) as error:  # a result: logged and kept
            self.controller.log("heal-failed", devices=list(devices), reason=str(error))
            return
        self.controller.log("healed", devices=list(devices), report=report)


def _tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def run(config: LiveConfig, scenario: str, network: str, nodes: int = 2) -> None:
    """Play ``scenario`` on ``network`` against the swarm, then the baseline."""
    endpoint = Endpoint(RUNNER)
    host, port = endpoint.address
    launch = ["srun", f"--nodes={nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1"]
    launch += [f"--gres=gpu:{DEVICES // nodes}", "--kill-on-bad-exit=0"]
    launch += [sys.executable, "-m", "syncytium.live.host", "--runner", f"{host}:{port}"]
    process = subprocess.Popen(launch)
    try:
        hosts = Hosts(endpoint, nodes)
        for system in SYSTEMS:
            out = config.output / f"{scenario}-{network}" / system
            Scenario(config, hosts, system, (scenario, network), out).play()
        hosts.stop()
    finally:
        process.wait(timeout=120)
        endpoint.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare, run or check H18's live scenarios."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=("prepare", "reference", "run", "check"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--network", choices=NETWORKS, action="append")
    parser.add_argument("--device", default="cuda", help="for prepare and reference")
    args = parser.parse_args(argv)
    config = LiveConfig.load(args.config)
    if args.command == "prepare":
        prepare(config, args.device)
    elif args.command == "reference" and not (config.output / "reference.pt").exists():
        from syncytium.live.checks import write_reference

        write_reference(config, args.device)
    elif args.command == "check":
        from syncytium.live.checks import write_checks

        write_checks(config)
    elif args.command == "run":
        append_run_metadata(config.output / "runs.jsonl", args.config)
        for scenario in args.scenario or sorted(SCENARIOS):
            for network in args.network or NETWORKS:
                run(config, scenario, network)


if __name__ == "__main__":
    main()
