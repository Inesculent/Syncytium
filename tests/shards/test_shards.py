import pytest
import torch

from holograph.eval import ParameterCounts, child_footprint
from holograph.models import split_parameters
from holograph.quant import Scheme
from holograph.shards import (
    MANIFEST_NAME,
    Manifest,
    ShardError,
    child_state_dict,
    read_child,
    rebuild_state_dict,
    shard_name,
    write_children,
)
from holograph.shards.writer import quantize_tensor

SCHEME = Scheme(bits=3, group_size=32, num_children=4)


def write(model, directory, scheme=SCHEME):
    return write_children(*split_parameters(model), scheme, "tiny", directory)


def read_all(directory, manifest, indices=None):
    indices = range(manifest.scheme.num_children) if indices is None else indices
    return [read_child(directory / shard_name(k), manifest, k) for k in indices]


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
    rebuilt = rebuild_state_dict(read_all(directory, manifest), manifest)
    for name, weight in split_parameters(tiny_model)[0].items():
        grid, _ = quantize_tensor(weight, SCHEME)
        error = (rebuilt[name] - weight.float()).reshape(-1, SCHEME.group_size).abs()
        assert torch.all(error <= grid.step / 8 * (1 + 1e-5))


def test_partial_rebuild_loses_some_precision(tiny_model, written):
    directory, manifest = written
    name = "model.embed_tokens.weight"
    original = dict(tiny_model.named_parameters())[name].float()
    full = rebuild_state_dict(read_all(directory, manifest), manifest)[name]
    partial = rebuild_state_dict(read_all(directory, manifest, [0, 1, 3]), manifest)[name]
    assert (partial - original).abs().mean() > (full - original).abs().mean()


def test_rebuilt_weights_load_into_the_model(tiny_model, written):
    directory, manifest = written
    state = rebuild_state_dict(read_all(directory, manifest), manifest)
    tiny_model.load_state_dict(state, strict=False)
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
    rebuilt = rebuild_state_dict(read_all(tmp_path, manifest), manifest)
    original = dict(tiny_model.named_parameters())["model.embed_tokens.weight"].float()
    assert torch.allclose(rebuilt["model.embed_tokens.weight"], original, atol=0.02)


def test_tampered_shard_is_rejected(written):
    directory, manifest = written
    path = directory / shard_name(1)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ShardError, match="does not match"):
        read_child(path, manifest, 1)


def test_shard_read_as_the_wrong_child_is_rejected(written):
    directory, manifest = written
    with pytest.raises(ShardError, match="child 2"):
        read_child(directory / shard_name(1), manifest, 2)


def test_inconsistent_parent_hash_is_rejected(written):
    directory, manifest = written
    forged = Manifest(**{**manifest.__dict__, "parent_sha256": "0" * 64})
    with pytest.raises(ShardError, match="parent"):
        rebuild_state_dict(read_all(directory, forged), forged)


def test_repeated_children_are_rejected(written):
    directory, manifest = written
    child = read_all(directory, manifest, [0])[0]
    with pytest.raises(ShardError, match="repeated"):
        rebuild_state_dict([child, child], manifest)


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
