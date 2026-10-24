import copy
import dataclasses
import json
import threading
from pathlib import Path

import pytest
import torch

from syncytium.live import host, run
from syncytium.live.checks import write_checks
from syncytium.live.device import CrashedError, device_from_args
from syncytium.live.pipeline import write_stages
from syncytium.live.run import LiveConfig, Scenario
from syncytium.models import split_parameters
from syncytium.quant import Scheme
from syncytium.shards import (
    MANIFEST_NAME,
    Manifest,
    Redundancy,
    rebuild_from_files,
    shard_name,
    write_children,
    write_regrowth,
)
from syncytium.tasks import gsm8k

CHILDREN = Scheme(bits=3, group_size=32, num_children=4, phase=0.5, rotation_seed=0, scaling=True)
RTN = Scheme(bits=6, group_size=32, num_children=1, phase=0.5, rotation_seed=0, scaling=True)
PROBLEMS = [gsm8k.Problem(f"What is {i} + {i}?", str(2 * i)) for i in range(12)]


def crash():
    raise CrashedError


class ThreadHosts:
    """Hosts that run each device the runner asks for as a thread, from the runner's argv."""

    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config

    def call(self, node, kind, info):
        if kind == "spawn":
            device = device_from_args(
                info["argv"][3:], tokenizer=self.tokenizer, model_config=self.config, crash=crash
            )
            threading.Thread(target=device.run, daemon=True).start()
            return {"pid": 0}
        return {"provision": host._provision, "clear": host._clear}[kind](info)

    def gpu(self, node, slot):
        return f"{node}:{slot}"


def write_set(model, scheme, directory):
    generator = torch.Generator().manual_seed(0)
    quantized, full = split_parameters(model)
    scales = {
        name: 0.5 + torch.rand(w.shape[-1], generator=generator)
        for name, w in quantized.items()
        if name != "model.embed_tokens.weight"
    }
    write_children(quantized, full, scheme, "tiny", directory, scales=scales)


@pytest.fixture
def config(tiny_model, tiny_config, tmp_path):
    children, baseline = tmp_path / "children", tmp_path / "rtn6"
    write_set(tiny_model, CHILDREN, children)
    write_regrowth(children, Redundancy.NEIGHBOUR)
    write_regrowth(children, Redundancy.RESIDUE, tolerated=2)
    write_set(tiny_model, RTN, baseline)
    write_stages(baseline, 2, tiny_config)
    return LiveConfig(
        name="live-tiny",
        output=tmp_path / "out",
        model="tiny",
        children=children,
        baseline=baseline,
        storage=tmp_path / "storage",
        group_size=32,
        timeout=1.0,
        max_new_tokens=8,
        per_phase=2,
        probe_problems=2,
        die_after=4,
        device="cpu",
        listen="127.0.0.1",
    )


def write_reference(config, tiny_model, tokenizer):
    """The simulation's probe: the parent's weights in one model."""
    manifest = Manifest.load(config.children / MANIFEST_NAME)
    parent = rebuild_from_files({k: config.children / shard_name(k) for k in range(4)}, manifest)
    model = copy.deepcopy(tiny_model)
    model.load_state_dict(parent, strict=False)
    prompts = []
    for problem in PROBLEMS[: config.probe_problems]:
        ids = torch.tensor([tokenizer.encode(gsm8k.prompt(problem.question))])
        with torch.no_grad():
            logits = model(ids).logits[0]
        prompts.append({"top1": logits.argmax(dim=-1).tolist(), "tail": logits[-16:]})
    config.output.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": prompts}, config.output / "reference.pt")


def test_every_scenario_runs_and_passes_its_checks(
    config, tiny_model, tiny_config, tokenizer, monkeypatch
):
    monkeypatch.setattr(run, "_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(
        run, "problems", lambda c: (PROBLEMS[: c.probe_problems], PROBLEMS[c.probe_problems :])
    )
    hosts = ThreadHosts(tokenizer, tiny_config)
    for scenario in ("A", "B", "C"):
        for system in run.SYSTEMS:
            out = config.output / f"{scenario}-cluster" / system
            Scenario(config, hosts, system, (scenario, "cluster"), out).play()
    write_reference(config, tiny_model, tokenizer)

    result = write_checks(config)
    for name, outcome in result["runs"].items():
        checks = outcome["checks"]
        # The devices run on the CPU, so no GPU memory is reported and (f) cannot pass here.
        failed = [c for c in "abcde" if not checks[c]["passed"]]
        assert not failed, (name, {c: checks[c] for c in failed})
        assert (config.output / f"timeline-{name}.png").exists()
    phases = result["runs"]["B-cluster"]["phases"]["baseline"]
    assert phases["P3"]["answered"] == 0 and phases["P1"]["answered"] == 2
    saved = json.loads((config.output / "checks.json").read_text())
    assert set(saved["runs"]) == {"A-cluster", "B-cluster", "C-cluster"}


@pytest.mark.parametrize("size", ["4b", "9b"])
def test_the_live_configs_load(size):
    config = LiveConfig.load(Path(f"configs/live-{size}.toml"))
    assert config.name == f"live-{size}" and config.lan == (0.0002, 1e9)
    assert (config.bits, config.group_size, config.die_after) == (3, 64, 32)


def test_a_set_written_with_another_scheme_is_written_again(tiny_model, tmp_path):
    # The first 4B sets stored grids in fp16; preparing again must replace them.
    config = LiveConfig.load(Path("configs/live-4b.toml"))
    live = run.live_scheme(3, 4, config)
    assert live.grid_storage == "float32" and live.scaling and live.group_size == 64
    scheme = dataclasses.replace(live, group_size=32)  # the tiny model has 32-wide weights
    write_set(tiny_model, scheme, tmp_path)
    assert run._written(tmp_path, scheme)
    assert not run._written(tmp_path, dataclasses.replace(scheme, grid_storage="float16"))
    assert not run._written(tmp_path / "missing", scheme)


def test_a_rerun_sets_the_earlier_attempt_aside(tmp_path):
    earlier = tmp_path / "A-cluster"
    (earlier / "swarm").mkdir(parents=True)
    (earlier / "swarm" / "events.jsonl").write_text("{}\n")
    assert run.set_aside(tmp_path / "B-lan") is None
    moved = run.set_aside(earlier)
    assert not earlier.exists() and moved.parent == tmp_path / run.REPLACED
    assert (moved / "swarm" / "events.jsonl").read_text() == "{}\n"
