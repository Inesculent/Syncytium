import copy
import json
import shutil
import threading

import pytest
import torch

from syncytium.live.device import CONTROLLER, CrashedError, DeviceSettings
from syncytium.live.packed import Part, load_member, part_names
from syncytium.live.pipeline import (
    STAGES_NAME,
    PipelineController,
    StageDevice,
    load_stage,
    stage_bounds,
    stage_inputs,
    stage_name,
    write_stages,
)
from syncytium.live.wire import Endpoint
from syncytium.models import split_parameters
from syncytium.quant import Scheme
from syncytium.shards import MANIFEST_NAME, SCALES_NAME, Manifest, write_children
from syncytium.tasks.generation import Decoding, generate

# The baseline's 6-bit round-to-nearest model: one "child" with offset ½.
RTN = Scheme(bits=6, group_size=32, num_children=1, phase=0.5, rotation_seed=0, scaling=True)
ROLES = {0: 0, 1: 1, 2: 0, 3: 1}
PROMPT = "Q: 2 + 2?\nA:"


def crash():
    raise CrashedError


@pytest.fixture
def reference(tiny_model, tmp_path):
    generator = torch.Generator().manual_seed(0)
    quantized, full = split_parameters(tiny_model)
    scales = {
        name: 0.5 + torch.rand(w.shape[-1], generator=generator)
        for name, w in quantized.items()
        if name != "model.embed_tokens.weight"
    }
    directory = tmp_path / "reference"
    write_children(quantized, full, RTN, "tiny", directory, scales=scales)
    return directory


def test_stage_bounds_cover_every_layer_once():
    assert stage_bounds(4, 2) == [(0, 2), (2, 4)]
    assert stage_bounds(5, 2) == [(0, 3), (3, 5)]
    assert stage_bounds(32, 3) == [(0, 11), (11, 22), (22, 32)]


def test_stages_hold_only_what_they_read(tiny_config, reference):
    manifest = Manifest.load(reference / MANIFEST_NAME)
    first = part_names(tiny_config, manifest, Part(0, 0, 2))
    last = part_names(tiny_config, manifest, Part(0, 2, 4))
    assert not any(".layers.2." in n or ".layers.3." in n for n in first)
    assert not any(".layers.0." in n or ".layers.1." in n for n in last)
    assert "model.norm.weight" in last and "model.norm.weight" not in first
    # The head is tied to the embedding, so both stages read it.
    assert "model.embed_tokens.weight/codes" in first and "model.embed_tokens.weight/codes" in last

    # Untied, the last stage needs no embedding; neither stage needs the other's head.
    untied = copy.deepcopy(tiny_config)
    untied.tie_word_embeddings = False
    assert "model.embed_tokens.weight/codes" not in part_names(untied, manifest, Part(0, 2, 4))


def test_two_stages_compute_the_whole_model(tiny_config, reference):
    write_stages(reference, 2, tiny_config)
    whole = load_member(reference, 0, "cpu", tiny_config)
    first = load_stage(reference, 0, "cpu", tiny_config)
    last = load_stage(reference, 1, "cpu", tiny_config)
    ids = torch.randint(2, 256, (1, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = whole(ids).logits
        hidden = first.model(input_ids=ids, **stage_inputs(first, 0, 10)).last_hidden_state
        actual = last(inputs_embeds=hidden, **stage_inputs(last, 0, 10)).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class Pipeline:
    """Four stage devices as threads, each with its own storage, and their controller."""

    def __init__(self, reference, root, tiny_config, tokenizer):
        self.root, self.config, self.tokenizer = root, tiny_config, tokenizer
        common = [MANIFEST_NAME, SCALES_NAME, STAGES_NAME]
        for k, stage in ROLES.items():
            self.storage(k).mkdir(parents=True)
            for name in [*common, stage_name(stage)]:
                shutil.copy(reference / name, self.storage(k) / name)
        self.endpoint = Endpoint(CONTROLLER, host="127.0.0.1")
        self.log = root / "events.jsonl"
        self.controller = PipelineController(
            self.endpoint, reference, tokenizer, self.log, ROLES, max_new_tokens=16
        )

    def storage(self, k):
        return self.root / f"device-{k}"

    def start(self, *indices):
        for k in indices:
            self.storage(k).mkdir(parents=True, exist_ok=True)
            settings = DeviceSettings(
                k, self.storage(k), self.endpoint.address, "cpu", timeout=1.0, stage=ROLES[k]
            )
            endpoint = Endpoint(k, host="127.0.0.1")
            device = StageDevice(settings, endpoint, self.tokenizer, self.config, crash)
            threading.Thread(target=device.run, daemon=True).start()
        self.controller.await_devices(indices, timeout=30)

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def close(self):
        self.controller.stop()
        self.endpoint.close()


@pytest.fixture
def pipeline(reference, tmp_path, tiny_config, tokenizer):
    write_stages(reference, 2, tiny_config)
    pipeline = Pipeline(reference, tmp_path / "devices", tiny_config, tokenizer)
    yield pipeline
    pipeline.close()


def test_the_pipeline_answers_as_the_whole_model(pipeline, reference, tiny_config, tokenizer):
    pipeline.start(0, 1, 2, 3)
    assert pipeline.controller.admit([0, 1, 2, 3]) == dict.fromkeys(range(4), True)
    record = pipeline.controller.serve(PROMPT)
    whole = load_member(reference, 0, "cpu", tiny_config)
    expected = generate(whole, tokenizer, [tokenizer.encode(PROMPT)], Decoding(16, ("\n\n", "Q:")))
    assert record["route"] == "pipeline" and record["members"] == [0, 1]
    assert record["text"] == expected[0].text


def test_the_pipeline_is_down_once_a_stage_has_no_copy(pipeline):
    controller = pipeline.controller
    pipeline.start(0, 1, 2, 3)
    controller.admit([0, 1, 2, 3])

    # A copy of stage 1 dies mid-answer; the request is retried through the other copy.
    record = controller.serve(PROMPT, force={"victim": 2, "die_after": 4})
    assert record["members"] == [0, 1] and record["text"] is not None
    controller.await_death(2, timeout=10)

    controller.kill([0], destroy=True)
    assert not controller.up()
    assert controller.serve(PROMPT)["route"] == "down"

    # A new device for stage 1 has nothing to copy from.
    pipeline.start(0)
    assert controller.restore([0])["devices"][0] == {"restored": False}
    assert [e["event"] for e in pipeline.events()].count("restore-impossible") == 1

    # The crashed copy returns with its storage; the new device copies from it.
    pipeline.start(2)
    assert controller.admit([2]) == {2: True}
    report = controller.restore([0])["devices"][0]
    assert report["restored"] and report["matches"] and report["bytes"] > 0
    assert controller.up() and controller.serve(PROMPT)["route"] == "pipeline"
