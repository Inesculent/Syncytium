import json
import shutil
import threading

import pytest
import torch

from syncytium.live.controller import Controller
from syncytium.live.device import CONTROLLER, CrashedError, Device, DeviceSettings
from syncytium.live.wire import Endpoint
from syncytium.models import split_parameters
from syncytium.quant import Scheme
from syncytium.shards import (
    MANIFEST_NAME,
    SCALES_NAME,
    Redundancy,
    shard_name,
    write_children,
    write_regrowth,
)
from syncytium.shards.regrow import regrowth_manifest_name, regrowth_name

SCHEME = Scheme(bits=3, group_size=32, num_children=4, phase=0.5, rotation_seed=0, scaling=True)
COMMON = [MANIFEST_NAME, SCALES_NAME, *(regrowth_manifest_name(k) for k in Redundancy)]
PROMPTS = ["Q: 1 + 1?\nA:", "Q: two cats\nA:"]


def crash():
    raise CrashedError


@pytest.fixture
def reference(tiny_model, tmp_path):
    """A full shard set with both kinds of regrowth data, as the controller's reference."""
    generator = torch.Generator().manual_seed(0)
    quantized, full = split_parameters(tiny_model)
    scales = {
        name: 0.5 + torch.rand(w.shape[-1], generator=generator)
        for name, w in quantized.items()
        if name != "model.embed_tokens.weight"
    }
    directory = tmp_path / "reference"
    write_children(quantized, full, SCHEME, "tiny", directory, scales=scales)
    write_regrowth(directory, Redundancy.NEIGHBOUR)
    write_regrowth(directory, Redundancy.RESIDUE, tolerated=2)
    return directory


class Swarm:
    """Devices as threads on one machine, each with its own storage, and their controller."""

    def __init__(self, reference, root, tiny_config, tokenizer):
        self.root, self.config, self.tokenizer = root, tiny_config, tokenizer
        for k in range(4):
            storage = self.storage(k)
            storage.mkdir(parents=True)
            for name in [*COMMON, shard_name(k), *(regrowth_name(r, k) for r in Redundancy)]:
                shutil.copy(reference / name, storage / name)
        self.endpoint = Endpoint(CONTROLLER, host="127.0.0.1")
        self.log = root / "events.jsonl"
        self.controller = Controller(
            self.endpoint, reference, tokenizer, self.log, max_new_tokens=24
        )
        self.threads = {}

    def storage(self, k):
        return self.root / f"device-{k}"

    def start(self, *indices):
        for k in indices:
            self.storage(k).mkdir(parents=True, exist_ok=True)
            settings = DeviceSettings(k, self.storage(k), self.endpoint.address, "cpu", timeout=1.0)
            endpoint = Endpoint(k, host="127.0.0.1")
            device = Device(settings, endpoint, self.tokenizer, self.config, crash)
            self.threads[k] = threading.Thread(target=device.run, daemon=True)
            self.threads[k].start()
        self.controller.await_devices(indices, timeout=30)

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def close(self):
        self.controller.stop()
        for thread in self.threads.values():
            thread.join(timeout=10)
        self.endpoint.close()


@pytest.fixture
def swarm(reference, tmp_path, tiny_config, tokenizer):
    swarm = Swarm(reference, tmp_path / "devices", tiny_config, tokenizer)
    yield swarm
    swarm.close()


def test_scenario_a_survives_and_heals_exactly(swarm):
    controller = swarm.controller
    swarm.start(0, 1, 2, 3)
    assert controller.admit([0, 1, 2, 3]) == dict.fromkeys(range(4), True)
    before = controller.probe(PROMPTS)
    assert before["consistent"] and before["members"] == [0, 1, 2, 3]
    assert controller.serve("Q: 3 + 4?\nA:")["text"] is not None

    # E1: device 2 crashes in the middle of a fused answer; the others finish it.
    forced = controller.serve("Q: 5 + 6?\nA:", force={"victim": 2, "die_after": 8})
    assert forced["route"] == "fused" and forced["text"] is not None
    assert forced["finished_with"] == [0, 1, 3]
    controller.await_death(2, timeout=10)
    assert controller.live() == [0, 1, 3]

    # E2: device 0 is destroyed, storage and all.
    controller.kill([0], destroy=True)
    assert controller.live() == [1, 3]
    assert not any(swarm.storage(0).iterdir())
    assert controller.serve("Q: 7 + 8?\nA:")["members"] == [1, 3]

    # E3: device 2 returns with its storage and is admitted by hash.
    swarm.start(2)
    assert controller.admit([2]) == {2: True}

    # E4: a new device with empty storage regrows child 0 from the survivors.
    swarm.start(0)
    report = controller.regrow([0])
    assert report["kind"] == "neighbour" and report["admitted"] == {0: True}
    assert report["devices"][0]["matches"] and report["devices"][0]["bytes"] > 0
    assert sorted(p.name for p in swarm.storage(0).iterdir()) == sorted(
        [*COMMON, shard_name(0), *(regrowth_name(r, 0) for r in Redundancy)]
    )

    after = controller.probe(PROMPTS)
    assert after["members"] == [0, 1, 2, 3]
    assert after["sha256"] == before["sha256"]


def test_adjacent_losses_refuse_neighbour_bits_and_regrow_from_the_residue(swarm):
    controller = swarm.controller
    swarm.start(0, 1, 2, 3)
    controller.admit([0, 1, 2, 3])
    before = controller.probe(PROMPTS[:1])
    controller.kill([0, 1], destroy=True)
    assert controller.serve("Q: 1 + 2?\nA:")["members"] == [2, 3]

    swarm.start(0, 1)
    report = controller.regrow([0, 1])
    assert report["kind"] == "residue" and report["admitted"] == {0: True, 1: True}
    refusals = [e for e in swarm.events() if e["event"] == "regrowth-refused"]
    assert [e["kind"] for e in refusals] == ["neighbour"]
    assert controller.probe(PROMPTS[:1])["sha256"] == before["sha256"]


def test_a_returning_device_with_a_wrong_shard_is_refused(swarm, reference):
    controller = swarm.controller
    swarm.start(0, 1, 2, 3)
    controller.admit([0, 1, 2, 3])
    controller.kill([3])
    path = swarm.storage(3) / shard_name(3)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    swarm.start(3)
    assert controller.admit([3]) == {3: False}
    assert controller.live() == [0, 1, 2]
