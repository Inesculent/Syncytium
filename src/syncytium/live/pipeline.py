"""The replicated pipeline baseline of H18 (``docs/preregistration-demo.md``).

The 6-bit round-to-nearest model (a single "child" with offset ½, as in the sweeps) is split
into two stages by layers, and each stage is held by two devices. A request goes through any
live copy of stage 1, which reads the prompt and drives decoding, to any live copy of stage 2,
which returns each next token. Each stage keeps its own cache, so every token costs one hidden
state forwards and one token back. A lost copy is restored only by copying its stage file from
a surviving copy: there is no other source, so once every copy of a stage is lost the pipeline
is down for good.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, PretrainedConfig

from syncytium.live.control import Control
from syncytium.live.device import Device
from syncytium.live.packed import Part, build_model, part_names
from syncytium.live.wire import Endpoint, Message
from syncytium.shards import MANIFEST_NAME, SCALES_NAME, Manifest, ShardError, read_scales
from syncytium.shards import shard_name as child_name
from syncytium.shards.manifest import file_sha256
from syncytium.tasks import gsm8k

STAGES_NAME = "stages.json"
# Stage 1 waits this many timeouts for a token before taking stage 2 for dead: stage 2 may
# be reading a long prompt.
TOKEN_PATIENCE = 6


@dataclass(frozen=True)
class Stage:
    """One stage of the pipeline.

    Attributes:
        first: Its first layer.
        last: One past its last layer.
        sha256: SHA-256 of its stage file.
    """

    first: int
    last: int
    sha256: str


def stage_name(stage: int) -> str:
    """File name of stage ``stage``'s tensors."""
    return f"stage-{stage}.safetensors"


def stage_bounds(layers: int, count: int) -> list[tuple[int, int]]:
    """``count`` runs of consecutive layers, as even as possible, earlier ones longer."""
    size, extra = divmod(layers, count)
    bounds, first = [], 0
    for stage in range(count):
        last = first + size + (stage < extra)
        bounds.append((first, last))
        first = last
    return bounds


def write_stages(
    directory: Path, count: int = 2, config: PretrainedConfig | None = None
) -> list[Stage]:
    """Split the single child of the set in ``directory`` into ``count`` stage files.

    Each stage file holds exactly the tensors its stage reads; :data:`STAGES_NAME` records
    the layers and hash of each.

    Raises:
        ShardError: If the set has more than one child.
    """
    manifest = Manifest.load(directory / MANIFEST_NAME)
    if manifest.scheme.num_children != 1:
        raise ShardError("a pipeline is cut from a set with a single child")
    config = config or AutoConfig.from_pretrained(manifest.model)
    layers = config.get_text_config().num_hidden_layers
    stages = []
    with safe_open(str(directory / child_name(0)), "pt") as file:
        for s, (first, last) in enumerate(stage_bounds(layers, count)):
            names = part_names(config, manifest, Part(0, first, last))
            path = directory / stage_name(s)
            save_file({name: file.get_tensor(name) for name in names}, str(path))
            stages.append(Stage(first, last, file_sha256(path)))
    record = {"model": manifest.model, "stages": [asdict(s) for s in stages]}
    (directory / STAGES_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return stages


def read_stages(directory: Path) -> list[Stage]:
    """The stages recorded in ``directory``."""
    raw = json.loads((directory / STAGES_NAME).read_text(encoding="utf-8"))
    return [Stage(**s) for s in raw["stages"]]


def load_stage(
    directory: Path, stage: int, device: str = "cuda", config: PretrainedConfig | None = None
) -> Any:
    """Build stage ``stage``'s model from its file, holding packed codes.

    Raises:
        ShardError: If the stage file does not match its recorded hash.
    """
    manifest = Manifest.load(directory / MANIFEST_NAME)
    record = read_stages(directory)[stage]
    path = directory / stage_name(stage)
    if file_sha256(path) != record.sha256:
        raise ShardError(f"{path} does not match the hash recorded for stage {stage}")
    scales = read_scales(directory / SCALES_NAME, manifest)
    return build_model(path, manifest, scales, Part(0, record.first, record.last), device, config)


def stage_inputs(model: Any, past: int, length: int) -> dict[str, Any]:
    """Positions (and, if needed, masks) for a stage reading ``length`` positions after ``past``.

    A stage's cache counts positions only in its full-attention layers, so positions are given
    explicitly. A stage with no full-attention layer gets its masks ready-made: one unpadded
    prompt needs none.
    """
    device = next(model.parameters()).device
    inputs: dict[str, Any] = {
        "position_ids": torch.arange(past, past + length, device=device)[None]
    }
    if "full_attention" not in model.config.layer_types:
        inputs["attention_mask"] = dict.fromkeys(("full_attention", "linear_attention"))
    return inputs


class StageDevice(Device):
    """A device holding one stage of the pipeline (``settings.stage``).

    A first-stage device answers requests, sending hidden states to a last-stage device and
    taking the next token back. A last-stage device serves hidden states from any first-stage
    device, one cache per request.
    """

    def handlers(self) -> dict[str, Callable[[Message], None]]:
        """Load, answer, copy a stage from another device, die and stop."""
        return {
            "load": self._load,
            "pipe": self._pipe,
            "copy": self._copy,
            "die": self._die,
            "stop": lambda _: self._stopped.set(),
        }

    def services(self) -> list[Callable[[], None]]:
        """A last-stage device serves hidden states."""
        return [self._serve_hidden]

    def _model_file(self) -> Path:
        return self.settings.storage / stage_name(self.settings.stage or 0)

    def _load(self, command: Message) -> None:
        path = self._model_file()
        actual = file_sha256(path) if path.exists() else None
        if actual != command.info["sha256"]:
            self._tell("refused", {"request": command.info["request"], "sha256": actual})
            return
        self.model = load_stage(
            self.settings.storage, self.settings.stage or 0, self.settings.device, self.model_config
        )
        self._ensure_tokenizer()
        self._warm_up(self._first_pass if self.settings.stage == 0 else self._last_pass)
        self._tell("ready", {"request": command.info["request"], "sha256": actual})

    def _first_pass(self, ids: torch.Tensor) -> Any:
        return self.model.model(input_ids=ids, **stage_inputs(self.model, 0, ids.shape[1]))

    def _last_pass(self, ids: torch.Tensor) -> Any:
        hidden = torch.zeros(*ids.shape, self.model.config.hidden_size, device=ids.device)
        hidden = hidden.to(_dtype(self.model))
        return self.model(inputs_embeds=hidden, **stage_inputs(self.model, 0, ids.shape[1]))

    def _pipe(self, command: Message) -> None:
        """Answer a request through stage 2 on device ``info["next"]``."""
        info, start = command.info, time.monotonic()
        tokens = self._decode(info)
        failed = tokens is None
        text = None if failed else gsm8k.truncate(self.tokenizer.decode(tokens))
        reply = {
            "request": info["request"],
            "failed": failed,
            "text": text,
            "tokens": 0 if failed else len(tokens),
            "seconds": time.monotonic() - start,
            "peak": self._peak(),
        }
        self._tell("answer", reply)

    def _decode(self, info: dict[str, Any]) -> list[int] | None:
        """Greedy answer tokens, or ``None`` if stage 2 stops answering."""
        target = (info["next"], tuple(info["next_address"]))
        ids = torch.tensor([info["prompt"]], device=self.settings.device)
        cache, past, tokens = None, 0, []
        try:
            while len(tokens) < info["max_new_tokens"]:
                if len(tokens) == info.get("die_after"):
                    self.crash()
                positions = stage_inputs(self.model, past, ids.shape[1])
                with torch.no_grad():
                    out = self.model.model(
                        input_ids=ids, past_key_values=cache, use_cache=True, **positions
                    )
                cache, past = out.past_key_values, past + ids.shape[1]
                token = self._next_token(target, info["request"], out.last_hidden_state)
                if token is None:
                    return None
                if token == self.tokenizer.eos_token_id:
                    break
                tokens.append(token)
                if any(stop in self.tokenizer.decode(tokens) for stop in gsm8k.STOPS):
                    break
                ids = torch.tensor([[token]], device=self.settings.device)
        finally:
            release = Message("release", self.index, info={"request": info["request"]})
            self.endpoint.send(*target, release)
        return tokens

    def _next_token(
        self, target: tuple[int, tuple[str, int]], request: int, hidden: Any
    ) -> int | None:
        info = {"request": request, "address": list(self.address)}
        if not self.endpoint.send(*target, Message("hidden", self.index, -1, info, hidden)):
            return None
        reply = self.endpoint.inbox.take(
            lambda m: m.kind == "token" and m.sender == target[0] and m.info["request"] == request,
            TOKEN_PATIENCE * self.settings.timeout,
        )
        return None if reply is None else reply.info["token"]

    def _serve_hidden(self) -> None:
        caches: dict[tuple[int, int], Any] = {}
        while not self._stopped.is_set():
            message = self.endpoint.inbox.take(lambda m: m.kind in ("hidden", "release"), 0.5)
            if message is None or self.model is None:
                continue
            key = (message.sender, message.info["request"])
            if message.kind == "release":
                caches.pop(key, None)
                continue
            hidden = message.tensor.to(self.settings.device)
            cache, past = caches.get(key, (None, 0))
            with torch.no_grad():
                out = self.model(
                    inputs_embeds=hidden,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    **stage_inputs(self.model, past, hidden.shape[1]),
                )
            caches[key] = (out.past_key_values, past + hidden.shape[1])
            token = int(out.logits[0, -1].argmax())
            reply = Message("token", self.index, info={"request": key[1], "token": token})
            self.endpoint.send(message.sender, tuple(message.info["address"]), reply)

    def _copy(self, command: Message) -> None:
        """Copy a stage file (and the set's common files) from a surviving copy."""
        start = time.monotonic()
        self.settings.storage.mkdir(parents=True, exist_ok=True)
        moved = self._fetch_all(command.info, self.settings.storage)
        info = {
            "request": command.info["request"],
            "sha256": file_sha256(self._model_file()),
            "bytes": moved,
            "seconds": time.monotonic() - start,
        }
        self._tell("copied", info)


class PipelineController(Control):
    """Serves requests through the replicated pipeline and restores copies by copying.

    Args:
        endpoint: The controller's endpoint.
        reference: A directory holding the set's manifest and :data:`STAGES_NAME`.
        tokenizer: The model's tokenizer.
        log: The JSON-lines event log to append to.
        roles: The stage each device holds, by device index.
        max_new_tokens: Most answer tokens per request.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        reference: Path,
        tokenizer: Any,
        log: Path,
        roles: Mapping[int, int],
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(endpoint, log)
        self.tokenizer, self.roles = tokenizer, dict(roles)
        self.stages = read_stages(reference)
        self.scaled = Manifest.load(reference / MANIFEST_NAME).scales_sha256 is not None
        self.max_new_tokens = max_new_tokens

    def expected_sha256(self, index: int) -> str:
        """The recorded hash of the stage device ``index`` holds."""
        return self.stages[self.roles[index]].sha256

    def copies(self, stage: int) -> list[int]:
        """Live devices holding ``stage``."""
        return [k for k in self.live() if self.roles[k] == stage]

    def up(self) -> bool:
        """Whether every stage has a live copy."""
        return all(self.copies(s) for s in range(len(self.stages)))

    def serve(
        self,
        prompt: str,
        force: dict[str, int] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer one request through live copies, retrying while copies die under it.

        ``force``, if given, routes through device ``force["victim"]`` (when it is a live
        first-stage copy), which crashes after ``force["die_after"]`` answer tokens. ``tags``
        are logged with the request.
        """
        ids, start, record = self.tokenizer.encode(prompt), time.monotonic(), None
        while record is None and self.up():
            record = self._attempt(ids, force)
            force = None
        record = record or {"route": "down", "members": [], "text": None}
        record |= {"seconds": time.monotonic() - start}
        record["prediction"] = gsm8k.extract(record["text"]) if record["text"] else None
        record |= tags or {}
        self.log("request", **record)
        return record

    def restore(self, indices: Sequence[int]) -> dict[str, Any]:
        """Give each new device in ``indices`` its stage by copying it from a live copy."""
        report: dict[str, Any] = {"devices": {}}
        for k in indices:
            stage = self.roles[k]
            sources = self.copies(stage)
            if not sources:
                self.log("restore-impossible", device=k, stage=stage)
                report["devices"][k] = {"restored": False}
                continue
            names = [stage_name(stage), MANIFEST_NAME, STAGES_NAME]
            names += [SCALES_NAME] if self.scaled else []
            info = {
                "files": {str(sources[0]): names},
                "sources": {str(sources[0]): list(self.devices[sources[0]].address)},
            }
            self.log("copying", device=k, stage=stage, source=sources[0])
            reply = self._reply(self._command(k, "copy", info), k, ("copied",))
            fields = reply.info if reply is not None else {"sha256": None}
            fields["matches"] = fields["sha256"] == self.expected_sha256(k)
            self.log("copied", device=k, stage=stage, source=sources[0], **fields)
            report["devices"][k] = fields | {"restored": self.admit([k])[k]}
        return report

    heal = restore

    def _attempt(self, ids: list[int], force: dict[str, int] | None) -> dict[str, Any] | None:
        first, last = self.copies(0)[0], self.copies(len(self.stages) - 1)[0]
        info = {
            "prompt": ids,
            "max_new_tokens": self.max_new_tokens,
            "next": last,
            "next_address": list(self.devices[last].address),
        }
        if force is not None and force["victim"] in self.copies(0):
            first = force["victim"]
            info["die_after"] = force["die_after"]
        reply = self._reply(self._command(first, "pipe", info), first, ("answer",))
        if reply is None or reply.info["failed"]:
            self.log("route-failed", members=[first, last])
            return None
        keep = ("request", "text", "tokens", "peak")
        return {"route": "pipeline", "members": [first, last]} | {k: reply.info[k] for k in keep}


def _dtype(model: Any) -> torch.dtype:
    return next(model.parameters()).dtype
