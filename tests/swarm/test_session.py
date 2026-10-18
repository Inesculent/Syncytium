import io
import math

import pytest
import torch

from syncytium.models import quantized_weights, split_parameters
from syncytium.quant import Scheme
from syncytium.shards import (
    Manifest,
    Redundancy,
    ShardError,
    rebuild_from_files,
    shard_name,
    verify_shard,
    write_children,
    write_regrowth,
)
from syncytium.swarm import Format, Swarm
from syncytium.swarm.cli import Console

# The tiny model's narrowest input dimension is 32, so groups of 32 fit every weight.
SCHEME = Scheme(bits=3, group_size=32, num_children=4, phase=0.5, rotation_seed=0)


@pytest.fixture
def swarm(tiny_model, tokenizer, tmp_path):
    write_children(*split_parameters(tiny_model), SCHEME, "tiny", tmp_path)
    write_regrowth(tmp_path, Redundancy.NEIGHBOUR)
    return Swarm(tmp_path, tiny_model, tokenizer, max_new_tokens=6)


def weights(swarm):
    return {n: w.detach().float().clone() for n, w in quantized_weights(swarm.model).items()}


def test_every_member_starts_alive(swarm):
    assert swarm.alive() == [0, 1, 2, 3]
    assert swarm.redundancy is Redundancy.NEIGHBOUR


def test_all_members_fused_are_the_rebuilt_parent(swarm, tmp_path):
    swarm.answer("2 + 2?", [0, 1, 2, 3], Format.RAW)
    manifest = Manifest.load(tmp_path / "manifest.json")
    parent = rebuild_from_files({k: tmp_path / shard_name(k) for k in range(4)}, manifest)
    for name, weight in weights(swarm).items():
        torch.testing.assert_close(
            weight, parent[name].to(torch.bfloat16).float(), atol=2e-2, rtol=0
        )


def test_a_lone_member_differs_from_the_fused_group(swarm):
    swarm.answer("hi", [2], Format.RAW)
    alone = weights(swarm)
    swarm.answer("hi", [0, 1, 2, 3], Format.RAW)
    fused = weights(swarm)
    assert any(not torch.equal(alone[n], fused[n]) for n in alone)


def test_an_unsure_leader_escalates_to_every_live_member(swarm):
    exchange = swarm.ask("hi", threshold=-1.0, form=Format.RAW)
    assert exchange.solo.members == (0,)
    assert exchange.fused.members == (0, 1, 2, 3)
    assert swarm.ask("hi", threshold=math.inf, form=Format.RAW).fused is None


def test_a_lost_member_cannot_answer_until_regrown(swarm, tmp_path):
    before = swarm.answer("2 + 3?", [1], Format.MATH).text
    moved = swarm.kill(1)
    assert [p.parent.name for p in moved] == ["lost", "lost"]
    assert swarm.alive() == [0, 2, 3]
    with pytest.raises(ShardError, match="not all alive"):
        swarm.answer("2 + 3?", [1], Format.MATH)
    assert swarm.ask("hi", threshold=-1.0, form=Format.RAW).fused.members == (0, 2, 3)
    assert swarm.regrowable() is None
    swarm.regrow()
    assert swarm.alive() == [0, 1, 2, 3]
    verify_shard(tmp_path / shard_name(1), swarm.manifest, 1)
    assert swarm.answer("2 + 3?", [1], Format.MATH).text == before


def test_adjacent_losses_cannot_be_regrown_with_neighbour_bits(swarm):
    swarm.kill(1)
    swarm.kill(2)
    assert "both lost" in swarm.regrowable()
    with pytest.raises(ShardError, match="both lost"):
        swarm.regrow()


def test_the_console_runs_piped_commands(swarm, monkeypatch, capsys):
    commands = "status\nask 1 + 1?\nkill 3\nfuse all 1 + 1?\nregrow\nsolo 9 hi\nnope\nquit\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(commands))
    Console(swarm, threshold=-1.0).cmdloop()
    out = capsys.readouterr().out
    assert "alive [0, 1, 2, 3]" in out
    assert "escalating" in out
    assert "[fused [0, 1, 2]]" in out
    assert "regrew child-3.safetensors: matches the manifest" in out
    assert "error: members [9] are not all alive" in out
    assert "unknown command: nope" in out
