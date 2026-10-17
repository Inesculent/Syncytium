import json

import pytest
import torch

from syncytium.eval import ParameterCounts, child_footprint
from syncytium.models import split_parameters
from syncytium.quant import Scheme
from syncytium.quant.rotation import random_signs
from syncytium.shards import (
    MANIFEST_NAME,
    Manifest,
    ShardError,
    child_state_dict,
    read_child,
    rebuild_from_files,
    shard_name,
    write_children,
)
from syncytium.shards import reader as reader_module
from syncytium.shards.timing import log_timing, time_rebuild
from syncytium.shards.writer import quantize_tensor

SCHEME = Scheme(bits=3, group_size=32, num_children=4)


def write(model, directory, scheme=SCHEME):
    return write_children(*split_parameters(model), scheme, "tiny", directory)


def read_all(directory, manifest, indices=None):
    indices = range(manifest.scheme.num_children) if indices is None else indices
    return [read_child(directory / shard_name(k), manifest, k) for k in indices]


def rebuild(directory, manifest, indices=None, **options):
    indices = range(manifest.scheme.num_children) if indices is None else indices
    return rebuild_from_files({k: directory / shard_name(k) for k in indices}, manifest, **options)


@pytest.fixture
def written(tiny_model, tmp_path):
    return tmp_path, write(tiny_model, tmp_path)


def test_writes_one_shard_per_child_and_a_manifest(written):
    directory, manifest = written
    names = {path.name for path in directory.iterdir()}
    assert names == {MANIFEST_NAME, *(shard_name(k) for k in range(4))}
    assert Manifest.load(directory / MANIFEST_NAME) == manifest


def test_read_codes_match_in_memory_quantization(tiny_model, written):
    directory, manifest = written
    name = "model.embed_tokens.weight"
    _, expected = quantize_tensor(dict(tiny_model.named_parameters())[name], SCHEME)
    for k, child in enumerate(read_all(directory, manifest)):
        assert torch.equal(child.codes[name], expected[k])


def test_full_rebuild_is_verified_and_within_the_fine_step(tiny_model, written):
    directory, manifest = written
    rebuilt = rebuild(directory, manifest)
    for name, weight in split_parameters(tiny_model)[0].items():
        grid, _ = quantize_tensor(weight, SCHEME)
        error = (rebuilt[name] - weight.float()).reshape(-1, SCHEME.group_size).abs()
        assert torch.all(error <= grid.step / 8 * (1 + 1e-5))


def test_partial_rebuild_loses_some_precision(tiny_model, written):
    directory, manifest = written
    name = "model.embed_tokens.weight"
    original = dict(tiny_model.named_parameters())[name].float()
    full = rebuild(directory, manifest)[name]
    partial = rebuild(directory, manifest, [0, 1, 3])[name]
    assert (partial - original).abs().mean() > (full - original).abs().mean()


def test_rebuilt_weights_load_into_the_model(tiny_model, written):
    directory, manifest = written
    state = rebuild(directory, manifest)
    missing, unexpected = tiny_model.load_state_dict(state, strict=False)
    assert not unexpected
    assert set(missing) <= {"lm_head.weight"}  # tied to the embeddings
    logits = tiny_model(torch.randint(0, 256, (1, 8))).logits
    assert torch.isfinite(logits).all()


def test_single_child_loads_as_a_model(tiny_model, written):
    directory, manifest = written
    (child,) = read_all(directory, manifest, [2])
    missing, unexpected = tiny_model.load_state_dict(
        child_state_dict(child, manifest), strict=False
    )
    assert not unexpected
    assert set(missing) <= {"lm_head.weight"}  # tied to the embeddings


def test_rotated_shards_rebuild_the_original(tiny_model, tmp_path):
    scheme = Scheme(bits=4, group_size=32, num_children=2, rotation_seed=7)
    manifest = write(tiny_model, tmp_path, scheme)
    rebuilt = rebuild(tmp_path, manifest)
    original = dict(tiny_model.named_parameters())["model.embed_tokens.weight"].float()
    assert torch.allclose(rebuilt["model.embed_tokens.weight"], original, atol=0.02)


def test_rotated_manifest_stores_the_signs(tiny_model, tmp_path):
    scheme = Scheme(bits=4, group_size=32, num_children=2, rotation_seed=7)
    manifest = write(tiny_model, tmp_path, scheme)
    assert manifest.rotation_signs == random_signs(32, 7)
    assert Manifest.load(tmp_path / MANIFEST_NAME).rotation_signs == manifest.rotation_signs
    assert write(tiny_model, tmp_path / "plain").rotation_signs is None


def test_rebuild_uses_the_stored_signs_not_the_seed(tiny_model, tmp_path):
    scheme = Scheme(bits=4, group_size=32, num_children=2, rotation_seed=7)
    manifest = write(tiny_model, tmp_path, scheme)
    expected = rebuild(tmp_path, manifest)
    reseeded = Manifest(**{**manifest.__dict__, "scheme": Scheme(4, 32, 2, rotation_seed=8)})
    rebuilt = rebuild(tmp_path, reseeded)
    assert all(torch.equal(rebuilt[name], weight) for name, weight in expected.items())


def test_rotated_manifest_without_signs_is_rejected(tiny_model, tmp_path):
    scheme = Scheme(bits=4, group_size=32, num_children=2, rotation_seed=7)
    manifest = write(tiny_model, tmp_path, scheme)
    with pytest.raises(ShardError, match="rotation signs"):
        Manifest(**{**manifest.__dict__, "rotation_signs": None})


def test_rebuild_in_bf16_is_the_float32_rebuild_rounded(written):
    directory, manifest = written
    full = rebuild(directory, manifest)
    half = rebuild(directory, manifest, dtype=torch.bfloat16)
    for spec in manifest.quantized():
        assert torch.equal(half[spec.name], full[spec.name].to(torch.bfloat16))


@pytest.mark.parametrize("indices", [None, [0, 1, 3]])
def test_chunking_does_not_change_the_rebuild(written, monkeypatch, indices):
    directory, manifest = written
    expected = rebuild(directory, manifest, indices)
    monkeypatch.setattr(reader_module, "CHUNK_GROUPS", 7)
    rebuilt = rebuild(directory, manifest, indices)
    assert all(torch.equal(rebuilt[name], weight) for name, weight in expected.items())


def test_tampered_shard_is_rejected(written):
    directory, manifest = written
    path = directory / shard_name(1)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ShardError, match="does not match"):
        read_child(path, manifest, 1)
    with pytest.raises(ShardError, match="does not match"):
        rebuild(directory, manifest)


def test_shard_read_as_the_wrong_child_is_rejected(written):
    directory, manifest = written
    with pytest.raises(ShardError, match="child 2"):
        read_child(directory / shard_name(1), manifest, 2)


def test_inconsistent_parent_hash_is_rejected(written):
    directory, manifest = written
    forged = Manifest(**{**manifest.__dict__, "parent_sha256": "0" * 64})
    with pytest.raises(ShardError, match="parent"):
        rebuild(directory, forged)


def test_rebuild_needs_a_child(written):
    _, manifest = written
    with pytest.raises(ShardError, match="at least one"):
        rebuild_from_files({}, manifest)


def test_repeated_children_are_rejected(written):
    directory, _ = written
    with pytest.raises(ShardError, match="repeated"):
        time_rebuild(directory, [0, 0])


def test_file_size_matches_the_memory_formula(tiny_model, written):
    directory, _ = written
    quantized, full = split_parameters(tiny_model)
    counts = ParameterCounts(
        quantized=sum(p.numel() for p in quantized.values()),
        full_precision=sum(p.numel() for p in full.values()),
    )
    expected = child_footprint(counts, SCHEME).total_bits / 8
    size = (directory / shard_name(0)).stat().st_size
    assert expected <= size <= expected + 16_384  # safetensors header and padding


def test_timed_rebuild_records_each_phase(written):
    directory, _ = written
    timing = time_rebuild(directory)
    assert timing.exact and timing.children == (0, 1, 2, 3) and timing.dtype == "bfloat16"
    assert timing.bytes_read == sum((directory / shard_name(k)).stat().st_size for k in range(4))
    assert timing.verify_seconds > 0 and timing.rebuild_seconds > 0
    partial = time_rebuild(directory, [0, 2])
    assert not partial.exact and partial.children == (0, 2)


def test_timings_accumulate_in_a_log(written):
    directory, _ = written
    for indices in (None, [1, 3]):
        log_timing(directory, time_rebuild(directory, indices))
    lines = (directory / "rebuilds.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["exact"] for r in records] == [True, False]
    assert all(r["total_seconds"] == r["verify_seconds"] + r["rebuild_seconds"] for r in records)
