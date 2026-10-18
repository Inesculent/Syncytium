"""A swarm you can query: real child shards on disk, run as agents in one process.

Each member is one child's shard file (written by ``syncytium-swarm create`` or
``syncytium-shard``) with its regrowth data. A member answers alone, or any live members fuse:
the fused weights are the mean of their children, which is what devices that average their
layer outputs compute, and with every member it is the exact parent. Killing a member moves
its files aside, as if its device were lost; regrowing rebuilds them from the survivors and
checks them against their hashes.

Weights are swapped into one model, as in the experiments, so the swarm shows what it
answers and how it degrades and heals, not how fast it would be on separate devices.
"""

import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel

from syncytium.models import WeightBank, load_model, load_tokenizer, quantized_weights
from syncytium.shards import (
    MANIFEST_NAME,
    Manifest,
    Redundancy,
    RegrowthManifest,
    ShardError,
    child_state_dict,
    read_child,
    regrow_shards,
    shard_name,
)
from syncytium.shards.regrow import check_losses, regrowth_manifest_name, regrowth_name
from syncytium.tasks import gsm8k
from syncytium.tasks.generation import Decoding, generate

LOST = "lost"  # where a lost member's files are moved, never read again


class Format(StrEnum):
    """How a question is put to the model."""

    MATH = "math"  # GSM8K's 8-shot "Q: … A:" prompt, as in the experiments
    RAW = "raw"  # the text as given, continued up to the first blank line


@dataclass(frozen=True)
class Reply:
    """One answer.

    Attributes:
        members: The children that produced it: one for a lone agent, several when fused.
        text: The answer.
        entropy: Mean entropy (nats) of the next-token distributions while answering.
        seconds: Time to load the weights and answer.
    """

    members: tuple[int, ...]
    text: str
    entropy: float
    seconds: float


@dataclass(frozen=True)
class Exchange:
    """A question answered by a lone agent, escalated to the fused group if it was unsure.

    Attributes:
        solo: The lone agent's answer.
        fused: The fused group's answer, or ``None`` if the agent was sure enough or alone.
        threshold: Entropy above which the agent escalated.
    """

    solo: Reply
    fused: Reply | None
    threshold: float


class Swarm:
    """The members of one shard set, each able to answer alone or fused with others.

    Args:
        directory: Directory holding the shards, their manifest and regrowth data.
        model: A model of the shards' architecture; its quantized weights are replaced.
        tokenizer: The model's tokenizer.
        max_new_tokens: Longest answer.
    """

    def __init__(
        self, directory: Path, model: PreTrainedModel, tokenizer: Any, max_new_tokens: int = 256
    ) -> None:
        self.directory, self.model, self.tokenizer = directory, model, tokenizer
        self.manifest = Manifest.load(directory / MANIFEST_NAME)
        self.max_new_tokens = max_new_tokens
        self.bank = WeightBank(quantized_weights(model))
        self._children: dict[int, dict[str, Tensor]] = {}
        self._loaded: tuple[int, ...] | None = None

    @classmethod
    def open(cls, directory: Path, device: str = "cpu", max_new_tokens: int = 256) -> "Swarm":
        """Open the swarm in ``directory`` with the model named in its manifest."""
        name = Manifest.load(directory / MANIFEST_NAME).model
        model = load_model(name, device=device, dtype=torch.bfloat16)
        return cls(directory, model, load_tokenizer(name), max_new_tokens)

    @property
    def size(self) -> int:
        """Number of members, N."""
        return self.manifest.scheme.num_children

    @property
    def redundancy(self) -> Redundancy | None:
        """Which regrowth data the set has, or ``None`` if it has none."""
        found = [k for k in Redundancy if (self.directory / regrowth_manifest_name(k)).exists()]
        return found[0] if found else None

    def alive(self) -> list[int]:
        """Members whose shard is present."""
        return [k for k in range(self.size) if (self.directory / shard_name(k)).exists()]

    def answer(self, question: str, members: Sequence[int], form: Format = Format.MATH) -> Reply:
        """Answer ``question`` with ``members`` fused (or alone, if there is one).

        Raises:
            ShardError: If a member is lost, or its shard does not match the manifest.
        """
        start = time.perf_counter()
        members = tuple(sorted(set(members)))
        self._load(members)
        prompt, stops = _prompt(question, form)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        [completion] = generate(
            self.model, self.tokenizer, [ids], Decoding(self.max_new_tokens, stops)
        )
        return Reply(
            members, completion.text.strip(), completion.entropy, time.perf_counter() - start
        )

    def ask(
        self,
        question: str,
        threshold: float,
        requester: int | None = None,
        form: Format = Format.MATH,
    ) -> Exchange:
        """Let one agent answer, and fuse every live member if its entropy exceeds ``threshold``.

        The requester defaults to the lowest live member, the deterministic leader.

        Raises:
            ShardError: If no member is alive or the requester is lost.
        """
        alive = self.alive()
        if not alive:
            raise ShardError("every member is lost; regrowth needs survivors")
        solo = self.answer(question, [alive[0] if requester is None else requester], form)
        fused = None
        if solo.entropy > threshold and len(alive) > 1:
            fused = self.answer(question, alive, form)
        return Exchange(solo, fused, threshold)

    def kill(self, index: int) -> list[Path]:
        """Lose member ``index``: move its shard and regrowth data to ``lost/``.

        Raises:
            ShardError: If ``index`` is not a live member.
        """
        if index not in self.alive():
            raise ShardError(f"member {index} is not alive")
        (self.directory / LOST).mkdir(exist_ok=True)
        files = [shard_name(index)]
        if self.redundancy is not None:
            files.append(regrowth_name(self.redundancy, index))
        moved = []
        for name in files:
            if (self.directory / name).exists():
                moved.append(Path(shutil.move(self.directory / name, self.directory / LOST / name)))
        self._children.pop(index, None)
        self._loaded = None
        return moved

    def regrowable(self) -> str | None:
        """Why the lost members cannot be regrown now, or ``None`` if they can."""
        lost = [k for k in range(self.size) if k not in self.alive()]
        if self.redundancy is None:
            return "this set has no regrowth data (write it with syncytium-swarm create)"
        sidecar = RegrowthManifest.load(self.directory / regrowth_manifest_name(self.redundancy))
        try:
            check_losses(lost, self.size, self.redundancy, sidecar.tolerated)
        except ShardError as error:
            return str(error)
        return None

    def regrow(self) -> list[Path]:
        """Regrow every lost member from the survivors; each file must match its hash.

        Raises:
            ShardError: If the set has no regrowth data, the losses are beyond what it
                survives, or a regrown file does not match its hash.
        """
        lost = [k for k in range(self.size) if k not in self.alive()]
        if not lost:
            return []
        if self.redundancy is None:
            raise ShardError("this set has no regrowth data")
        self._loaded = None
        return regrow_shards(self.directory, lost, self.redundancy)

    def _load(self, members: tuple[int, ...]) -> None:
        if members == self._loaded:
            return
        alive = self.alive()
        if not members or any(k not in alive for k in members):
            raise ShardError(f"members {list(members)} are not all alive (alive: {alive})")
        states = [self._child(k) for k in members]
        mean = {
            name: torch.stack([s[name].float() for s in states]).mean(dim=0) for name in states[0]
        }
        self.bank.load(mean)
        self._loaded = members

    def _child(self, index: int) -> dict[str, Tensor]:
        """Dequantized weights of member ``index``, read and verified once, kept in bf16."""
        if index not in self._children:
            path = self.directory / shard_name(index)
            child = read_child(path, self.manifest, index)
            state = child_state_dict(child, self.manifest)
            names = set(quantized_weights(self.model))
            self._children[index] = {
                n: t.to(torch.bfloat16) for n, t in state.items() if n in names
            }
        return self._children[index]


def _prompt(question: str, form: Format) -> tuple[str, tuple[str, ...]]:
    if form == Format.MATH:
        return gsm8k.prompt(question), gsm8k.STOPS
    return question, ("\n\n",)
