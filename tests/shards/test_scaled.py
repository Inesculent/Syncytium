import hashlib
import json
from functools import partial

import pytest
import torch
from safetensors.torch import load_file, save_file

from syncytium.models import input_moments, quantized_weights, split_parameters
from syncytium.quant import ChannelScales, Estimator, Scheme, simulate_child, simulate_rebuild
from syncytium.quant import simulate as simulate_module
from syncytium.quant.grid import fit_grid
from syncytium.shards import (
    MANIFEST_NAME,
    SCALES_NAME,
    Manifest,
    Redundancy,
    ShardError,
    child_state_dict,
    read_child,
    rebuild_from_files,
    regrow_shards,
    shard_name,
    verify_shard,
    write_children,
    write_regrowth,
)
from syncytium.shards import reader as reader_module
from syncytium.shards.calibration import weight_scales
from syncytium.shards.cli import main
from syncytium.shards.manifest import file_sha256, update_codes_hash
from syncytium.shards.writer import quantize_tensor

UNSCALED = "model.layers.0.linear_attn.in_proj_qkv.weight"  # left out of the synthetic scales


def scaled_scheme(rotation_seed=None, num_children=4):
    return Scheme(3, 32, num_children, 0.5, rotation_seed, scaling=True)


def synthetic_scales(model, spread=0.5):
    """Log-normal scales for every quantized weight but one, which stays unscaled."""
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.exp(spread * torch.randn(weight.shape[-1], generator=generator))
        for name, weight in quantized_weights(model).items()
        if name != UNSCALED
    }


def write(model, directory, scheme, scales):
    return write_children(*split_parameters(model), scheme, "tiny", directory, scales)


def paths(directory, indices):
    return {k: directory / shard_name(k) for k in indices}


def originals(model):
    return {name: w.detach().float() for name, w in quantized_weights(model).items()}


@pytest.fixture(params=[None, 7], ids=["unrotated", "rotated"])
def scheme(request):
    return scaled_scheme(request.param)


@pytest.fixture
def scales(tiny_model):
    return synthetic_scales(tiny_model)


@pytest.fixture
def written(tiny_model, scheme, scales, tmp_path):
    return tmp_path, write(tiny_model, tmp_path, scheme, scales)


@pytest.fixture
def stored_grid(monkeypatch):
    """Simulate on the grid as shards store it (m and Δ rounded to fp16), not the float32 one."""
    monkeypatch.setattr(simulate_module, "fit_grid", partial(fit_grid, storage=torch.float16))


def test_scales_are_stored_once_next_to_the_shards(written, scales):
    directory, manifest = written
    names = {path.name for path in directory.iterdir()}
    assert names == {MANIFEST_NAME, SCALES_NAME, *(shard_name(k) for k in range(4))}
    assert manifest.scales_sha256 == file_sha256(directory / SCALES_NAME)
    assert Manifest.load(directory / MANIFEST_NAME) == manifest
    stored = load_file(directory / SCALES_NAME)
    assert stored.keys() == scales.keys() and UNSCALED not in stored
    assert all(
        stored[n].dtype == torch.float32 and torch.equal(stored[n], scales[n]) for n in stored
    )


def test_codes_are_those_of_the_scaled_weight(tiny_model, written, scheme, scales):
    directory, manifest = written
    name = "model.embed_tokens.weight"
    weight = dict(tiny_model.named_parameters())[name]
    _, expected = quantize_tensor(
        weight * scales[name], Scheme(3, 32, 4, 0.5, scheme.rotation_seed)
    )
    child = read_child(directory / shard_name(1), manifest, 1)
    assert torch.equal(child.codes[name], expected[1])
    assert torch.equal(child.scales[name], scales[name])


def test_children_match_the_simulation(tiny_model, written, scheme, scales, stored_grid):
    directory, manifest = written
    for k in range(4):
        state = child_state_dict(read_child(directory / shard_name(k), manifest, k), manifest)
        for name, weight in originals(tiny_model).items():
            expected = simulate_child(weight, scheme, k, scales.get(name))
            torch.testing.assert_close(state[name], expected, rtol=1e-6, atol=1e-7)


def test_full_rebuild_matches_the_simulation(tiny_model, written, scheme, scales, stored_grid):
    directory, manifest = written
    rebuilt = rebuild_from_files(paths(directory, range(4)), manifest)
    for name, weight in originals(tiny_model).items():
        expected = simulate_rebuild(weight, scheme, range(4), scales=scales.get(name))
        torch.testing.assert_close(rebuilt[name], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("indices", [[0, 1, 3], [2]])
def test_partial_rebuild_matches_the_midpoint_simulation(
    tiny_model, written, scheme, scales, stored_grid, indices
):
    directory, manifest = written
    rebuilt = rebuild_from_files(paths(directory, indices), manifest)
    for name, weight in originals(tiny_model).items():
        expected = simulate_rebuild(weight, scheme, indices, Estimator.MIDPOINT, scales.get(name))
        torch.testing.assert_close(rebuilt[name], expected, rtol=1e-6, atol=1e-7)


def test_rebuild_is_close_to_the_original(tiny_model, written):
    directory, manifest = written
    rebuilt = rebuild_from_files(paths(directory, range(4)), manifest)
    original = originals(tiny_model)["model.embed_tokens.weight"]
    assert torch.allclose(rebuilt["model.embed_tokens.weight"], original, atol=0.02)


@pytest.mark.parametrize("indices", [range(4), [0, 1, 3]])
def test_chunking_does_not_change_the_scaled_rebuild(written, monkeypatch, indices):
    directory, manifest = written
    expected = rebuild_from_files(paths(directory, indices), manifest)
    monkeypatch.setattr(reader_module, "CHUNK_GROUPS", 7)  # chunks straddle weight rows
    rebuilt = rebuild_from_files(paths(directory, indices), manifest)
    for name, weight in expected.items():
        # Rotating back is a matmul whose rounding may depend on the chunk's size.
        torch.testing.assert_close(rebuilt[name], weight, rtol=1e-6, atol=1e-7)


def test_scales_are_divided_out_before_casting(written):
    directory, manifest = written
    full = rebuild_from_files(paths(directory, range(4)), manifest)
    half = rebuild_from_files(paths(directory, range(4)), manifest, dtype=torch.bfloat16)
    for spec in manifest.quantized():
        assert torch.equal(half[spec.name], full[spec.name].to(torch.bfloat16))


def test_unit_scales_keep_the_codes_but_the_hash_covers_the_scales(tiny_model, tmp_path):
    plain = write(tiny_model, tmp_path / "plain", Scheme(3, 32, 4, 0.5), None)
    ones = {n: torch.ones(w.shape[-1]) for n, w in quantized_weights(tiny_model).items()}
    scaled = write(tiny_model, tmp_path / "scaled", scaled_scheme(), ones)
    assert plain.scales_sha256 is None
    assert scaled.shard_sha256 == plain.shard_sha256  # identical codes, grids and tensors
    assert scaled.parent_sha256 != plain.parent_sha256
    codes = {k: read_child(tmp_path / "plain" / shard_name(k), plain, k).codes for k in range(4)}
    digest = hashlib.sha256()  # an unscaled set's parent hash covers its codes alone
    for spec in plain.quantized():
        update_codes_hash(digest, spec.name, sum(c[spec.name].int() for c in codes.values()))
    assert digest.hexdigest() == plain.parent_sha256


def test_a_manifest_without_a_scales_entry_still_loads(tiny_model, tmp_path):
    manifest = write(tiny_model, tmp_path, Scheme(3, 32, 4), None)
    raw = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    del raw["scales_sha256"]
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(raw), encoding="utf-8")
    assert Manifest.load(tmp_path / MANIFEST_NAME) == manifest


def test_tampered_scales_file_is_rejected(written):
    directory, manifest = written
    path = directory / SCALES_NAME
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ShardError, match="channel scales"):
        read_child(directory / shard_name(0), manifest, 0)
    for indices in (range(4), [0, 2]):
        with pytest.raises(ShardError, match="channel scales"):
            rebuild_from_files(paths(directory, indices), manifest)


def test_missing_scales_file_is_rejected(written):
    directory, manifest = written
    (directory / SCALES_NAME).unlink()
    with pytest.raises(ShardError, match="channel scales"):
        rebuild_from_files(paths(directory, [1]), manifest)


def test_swapped_scales_fail_the_parent_hash(written, scales):
    directory, manifest = written
    save_file({name: 2 * vector for name, vector in scales.items()}, directory / SCALES_NAME)
    forged = Manifest(
        **{**manifest.__dict__, "scales_sha256": file_sha256(directory / SCALES_NAME)}
    )
    with pytest.raises(ShardError, match="parent"):
        rebuild_from_files(paths(directory, range(4)), forged)


def test_manifest_has_a_scales_hash_exactly_when_the_scheme_scales(written):
    _, manifest = written
    with pytest.raises(ShardError, match="scales hash"):
        Manifest(**{**manifest.__dict__, "scales_sha256": None})
    unscaled = Scheme(3, 32, 4, 0.5, manifest.scheme.rotation_seed)
    with pytest.raises(ShardError, match="scales hash"):
        Manifest(**{**manifest.__dict__, "scheme": unscaled})


@pytest.mark.parametrize(
    ("scheme", "change", "message"),
    [
        (scaled_scheme(), lambda s: None, "needs the channel scales"),
        (Scheme(3, 32, 4), lambda s: s, "unscaled scheme"),
        (scaled_scheme(), lambda s: s | {"nope.weight": torch.ones(32)}, "not a quantized"),
        (scaled_scheme(), lambda s: s | {"lm_head.weight": torch.ones(32)}, "not a quantized"),
        (scaled_scheme(), lambda s: s | {"model.embed_tokens.weight": torch.ones(8)}, "one pos"),
        (scaled_scheme(), lambda s: s | {"model.embed_tokens.weight": torch.zeros(64)}, "one pos"),
        (
            scaled_scheme(),
            lambda s: s | {"model.embed_tokens.weight": torch.full((64,), torch.inf)},
            "finite",
        ),
    ],
)
def test_writer_checks_the_scales(tiny_model, scales, tmp_path, scheme, change, message):
    with pytest.raises(ValueError, match=message):
        write(tiny_model, tmp_path, scheme, change(scales))
    assert not (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.parametrize(("kind", "lost"), [(Redundancy.NEIGHBOUR, [2]), (Redundancy.RESIDUE, [0])])
def test_scaled_children_regrow_to_their_hashes(written, kind, lost):
    directory, manifest = written
    scales_hash = file_sha256(directory / SCALES_NAME)
    expected = rebuild_from_files(paths(directory, range(4)), manifest)
    write_regrowth(directory, kind)
    for index in lost:
        (directory / shard_name(index)).unlink()
    regrown = regrow_shards(directory, lost, kind)
    assert [p.name for p in regrown] == [shard_name(k) for k in lost]
    for index in range(4):
        verify_shard(directory / shard_name(index), manifest, index)
    assert file_sha256(directory / SCALES_NAME) == scales_hash
    rebuilt = rebuild_from_files(paths(directory, range(4)), manifest)
    assert all(torch.equal(rebuilt[name], weight) for name, weight in expected.items())


def test_regrowth_checks_the_scales_file(written, scales):
    directory, _ = written
    write_regrowth(directory, Redundancy.NEIGHBOUR)
    (directory / shard_name(1)).unlink()
    save_file({name: 2 * vector for name, vector in scales.items()}, directory / SCALES_NAME)
    with pytest.raises(ShardError, match="channel scales"):
        regrow_shards(directory, [1], Redundancy.NEIGHBOUR)
    assert not (directory / shard_name(1)).exists()


def calibration_batches():
    return torch.randint(2, 256, (4, 64), generator=torch.Generator().manual_seed(0))


def test_searched_scales_round_trip(tiny_model, tmp_path, stored_grid):
    scheme = scaled_scheme(rotation_seed=0)
    weights = quantized_weights(tiny_model)
    with torch.no_grad():
        searched = ChannelScales(input_moments(tiny_model, calibration_batches()))
        scales = weight_scales(searched, weights, scheme)
    assert scales.keys() == weights.keys()  # tied embeddings get the output head's moments
    assert any(not torch.equal(s, torch.ones_like(s)) for s in scales.values())
    manifest = write(tiny_model, tmp_path, scheme, scales)
    rebuilt = rebuild_from_files(paths(tmp_path, range(4)), manifest)
    for name, weight in originals(tiny_model).items():
        expected = simulate_rebuild(weight, scheme, range(4), scales=scales[name])
        torch.testing.assert_close(rebuilt[name], expected, rtol=1e-6, atol=1e-7)


@pytest.fixture
def fake_sources(tiny_model, tokenizer, monkeypatch):
    """Point the shard command line at the tiny model and random 'WikiText-2' train text."""
    from syncytium.shards import calibration

    seen = {}

    def corpus(_, split="test"):
        seen["split"] = split
        return torch.randint(2, 256, (3 * 512 + 10,), generator=torch.Generator().manual_seed(1))

    monkeypatch.setattr(calibration, "load_model", lambda *_, **__: tiny_model)
    monkeypatch.setattr(calibration, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(calibration, "load_wikitext2", corpus)
    return seen


def test_shard_command_line_calibrates_like_a_sweep(tiny_model, fake_sources, tmp_path):
    main(["tiny", "--out", str(tmp_path), "--bits", "3", "--group-size", "32", "--scaling"])
    manifest = Manifest.load(tmp_path / MANIFEST_NAME)
    assert manifest.scheme == Scheme(3, 32, 4, 0.0, None, scaling=True)
    assert fake_sources["split"] == "train"
    batches = torch.randint(2, 256, (3 * 512 + 10,), generator=torch.Generator().manual_seed(1))
    searched = ChannelScales(input_moments(tiny_model, batches[: 3 * 512].reshape(3, 512)))
    expected = weight_scales(searched, quantized_weights(tiny_model), manifest.scheme)
    stored = load_file(tmp_path / SCALES_NAME)
    assert stored.keys() == expected.keys()
    assert all(torch.equal(stored[name], expected[name].float()) for name in stored)
    rebuild_from_files(paths(tmp_path, range(4)), manifest)


def test_shard_command_line_without_scaling_writes_no_scales(fake_sources, tmp_path):
    main(["tiny", "--out", str(tmp_path), "--group-size", "32"])
    assert not (tmp_path / SCALES_NAME).exists()
    assert "split" not in fake_sources
