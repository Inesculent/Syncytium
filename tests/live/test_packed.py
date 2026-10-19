import copy
import threading

import pytest
import torch

from syncytium.live.exchange import Exchange
from syncytium.live.packed import load_member, packed_bytes, set_exchange, unpack_fast
from syncytium.live.wire import Endpoint
from syncytium.models import quantized_weights, split_parameters
from syncytium.quant import Scheme
from syncytium.shards import (
    MANIFEST_NAME,
    Manifest,
    child_state_dict,
    read_child,
    rebuild_from_files,
    shard_name,
    write_children,
)
from syncytium.shards.packing import pack, unpack

# The tiny model's narrowest input dimension is 32, so groups of 32 fit every weight.
SCHEME = Scheme(bits=3, group_size=32, num_children=4, phase=0.5, rotation_seed=0, scaling=True)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_fast_unpacking_matches_the_shard_format(bits):
    codes = torch.randint(0, 1 << bits, (4096,), generator=torch.Generator().manual_seed(bits))
    packed = pack(codes.to(torch.uint8), bits)
    assert torch.equal(unpack_fast(packed, bits, 0, 4096), unpack(packed, bits, 4096))
    assert torch.equal(unpack_fast(packed, bits, 1024, 512), codes[1024:1536].to(torch.uint8))
    with pytest.raises(ValueError, match="multiples of 8"):
        unpack_fast(packed, bits, 3, 8)


@pytest.fixture
def shards(tiny_model, tmp_path):
    generator = torch.Generator().manual_seed(0)
    quantized, full = split_parameters(tiny_model)
    scales = {
        name: 0.5 + torch.rand(w.shape[-1], generator=generator)
        for name, w in quantized.items()
        if name != "model.embed_tokens.weight"
    }
    write_children(quantized, full, SCHEME, "tiny", tmp_path, scales=scales)
    return tmp_path


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


IDS = torch.randint(2, 256, (1, 12), generator=torch.Generator().manual_seed(1))


def reference(tiny_model, state):
    model = copy.deepcopy(tiny_model)
    model.load_state_dict(state, strict=False)
    return model


def test_a_member_built_from_its_shard_is_that_child(tiny_model, tiny_config, shards):
    manifest = Manifest.load(shards / MANIFEST_NAME)
    for k in (0, 3):
        member = load_member(shards, k, "cpu", tiny_config)
        child = read_child(shards / shard_name(k), manifest, k)
        expected = logits(reference(tiny_model, child_state_dict(child, manifest)), IDS)
        torch.testing.assert_close(logits(member, IDS), expected, rtol=1e-4, atol=1e-4)


def test_a_member_holds_only_packed_codes(tiny_model, tiny_config, shards):
    member = load_member(shards, 1, "cpu", tiny_config)
    full = sum(w.numel() * 4 for w in quantized_weights(tiny_model).values())
    assert packed_bytes(member) < full / 2
    assert not any(isinstance(m, torch.nn.Linear) for m in member.modules())


def test_four_members_fused_over_sockets_are_the_parent(tiny_model, tiny_config, shards):
    manifest = Manifest.load(shards / MANIFEST_NAME)
    parent = rebuild_from_files({k: shards / shard_name(k) for k in range(4)}, manifest)
    expected = logits(reference(tiny_model, parent), IDS)
    members = [load_member(shards, k, "cpu", tiny_config) for k in range(4)]
    endpoints = {k: Endpoint(k, host="127.0.0.1") for k in range(4)}
    addresses = {k: ("127.0.0.1", e._server.getsockname()[1]) for k, e in endpoints.items()}
    outputs = {}

    def run(k):
        exchange = Exchange(endpoints[k], addresses, range(4), timeout=5)
        set_exchange(members[k], exchange.average)
        outputs[k] = logits(members[k], IDS)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for e in endpoints.values():
        e.close()
    assert all(torch.equal(outputs[k], outputs[0]) for k in range(4))
    torch.testing.assert_close(outputs[0], expected, rtol=1e-3, atol=1e-3)
