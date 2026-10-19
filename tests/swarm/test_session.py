import io
import math
from functools import partial

import pytest
import torch

from syncytium.models import quantized_weights, split_parameters
from syncytium.quant import Scheme, simulate_rebuild
from syncytium.quant import simulate as simulate_module
from syncytium.quant.grid import fit_grid
from syncytium.shards import (
    SCALES_NAME,
    Manifest,
    Redundancy,
    ShardError,
    child_state_dict,
    read_child,
    rebuild_from_files,
    shard_name,
    verify_shard,
    write_children,
    write_regrowth,
)
from syncytium.swarm import Format, Swarm
from syncytium.swarm.cli import Console, main

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


# Scaled swarms: the members' scales live in one file next to their shards.

SCALED = Scheme(bits=3, group_size=32, num_children=4, phase=0.5, rotation_seed=0, scaling=True)


def parent_of(directory):
    manifest = Manifest.load(directory / "manifest.json")
    return rebuild_from_files({k: directory / shard_name(k) for k in range(4)}, manifest)


def assert_fused_is_parent(swarm, directory):
    swarm.answer("2 + 2?", [0, 1, 2, 3], Format.RAW)
    parent = parent_of(directory)
    for name, weight in weights(swarm).items():
        torch.testing.assert_close(
            weight, parent[name].to(torch.bfloat16).float(), atol=2e-2, rtol=0
        )


def synthetic_scales(model):
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.exp(0.5 * torch.randn(w.shape[-1], generator=generator))
        for name, w in quantized_weights(model).items()
    }


@pytest.fixture
def scaled_swarm(tiny_model, tokenizer, tmp_path):
    scales = synthetic_scales(tiny_model)
    write_children(*split_parameters(tiny_model), SCALED, "tiny", tmp_path, scales)
    write_regrowth(tmp_path, Redundancy.NEIGHBOUR)
    return Swarm(tmp_path, tiny_model, tokenizer, max_new_tokens=6)


def test_scaled_members_fuse_to_the_rebuilt_parent(scaled_swarm, tmp_path, monkeypatch):
    original = weights(scaled_swarm)
    assert_fused_is_parent(scaled_swarm, tmp_path)
    # Simulate on the grid as shards store it (fp16), so only bf16 rounding remains.
    monkeypatch.setattr(simulate_module, "fit_grid", partial(fit_grid, storage=torch.float16))
    scales = synthetic_scales(scaled_swarm.model)
    for name, weight in weights(scaled_swarm).items():
        expected = simulate_rebuild(original[name], SCALED, range(4), scales=scales[name])
        torch.testing.assert_close(weight, expected, rtol=1e-2, atol=1e-3)


def test_a_scaled_member_alone_is_its_unscaled_child(scaled_swarm, tmp_path):
    scaled_swarm.answer("hi", [2], Format.RAW)
    manifest = scaled_swarm.manifest
    child = child_state_dict(read_child(tmp_path / shard_name(2), manifest, 2), manifest)
    for name, weight in weights(scaled_swarm).items():
        assert torch.equal(weight, child[name].to(torch.bfloat16).float())


def test_a_lost_scaled_member_regrows_and_keeps_its_scales(scaled_swarm, tmp_path):
    before = scaled_swarm.answer("2 + 3?", [3], Format.MATH).text
    scaled_swarm.kill(3)
    assert (tmp_path / SCALES_NAME).exists()  # shared by the set, not lost with a member
    scaled_swarm.regrow()
    verify_shard(tmp_path / shard_name(3), scaled_swarm.manifest, 3)
    assert scaled_swarm.answer("2 + 3?", [3], Format.MATH).text == before


def test_create_with_scaling_writes_a_scaled_swarm(tiny_model, tokenizer, monkeypatch, tmp_path):
    from syncytium.shards import calibration

    corpus = torch.randint(2, 256, (2 * 512,), generator=torch.Generator().manual_seed(0))
    monkeypatch.setattr(calibration, "load_model", lambda *_, **__: tiny_model)
    monkeypatch.setattr(calibration, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(calibration, "load_wikitext2", lambda *_, **__: corpus)
    main(["create", "tiny", str(tmp_path), "--group-size", "32", "--scaling"])
    manifest = Manifest.load(tmp_path / "manifest.json")
    assert manifest.scheme == SCALED and manifest.scales_sha256 is not None
    assert (tmp_path / SCALES_NAME).exists()
    swarm = Swarm(tmp_path, tiny_model, tokenizer, max_new_tokens=6)
    assert swarm.redundancy is Redundancy.NEIGHBOUR
    assert_fused_is_parent(swarm, tmp_path)
