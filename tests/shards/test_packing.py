import pytest
import torch

from syncytium.shards import pack, packed_size, unpack
from syncytium.shards import packing as packing_module


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("count", [0, 1, 7, 8, 1001])
def test_round_trip(bits, count):
    codes = torch.randint(0, 2**bits, (count,), generator=torch.Generator().manual_seed(bits))
    packed = pack(codes.to(torch.uint8), bits)
    assert packed.numel() == packed_size(count, bits)
    assert torch.equal(unpack(packed, bits, count), codes.to(torch.uint8))


@pytest.mark.parametrize("bits", range(1, 9))
def test_unpacking_in_chunks_gives_the_same_codes(bits, monkeypatch):
    codes = torch.randint(0, 2**bits, (1001,), generator=torch.Generator().manual_seed(bits))
    packed = pack(codes.to(torch.uint8), bits)
    monkeypatch.setattr(packing_module, "CHUNK_CODES", 16)
    assert torch.equal(unpack(packed, bits, 1001), codes.to(torch.uint8))


def test_three_bit_layout_is_least_significant_bit_first():
    # Codes 1, 2, 3 give the bit stream 100 010 110, i.e. bytes 0b11010001 and 0b0.
    packed = pack(torch.tensor([1, 2, 3], dtype=torch.uint8), bits=3)
    assert packed.tolist() == [0b11010001, 0]


def test_multidimensional_codes_are_flattened():
    codes = torch.tensor([[1, 2], [3, 0]], dtype=torch.uint8)
    assert torch.equal(unpack(pack(codes, 2), 2, 4), codes.reshape(-1))


def test_rejects_codes_that_do_not_fit():
    with pytest.raises(ValueError, match="do not fit in 2 bits"):
        pack(torch.tensor([4], dtype=torch.uint8), bits=2)


def test_rejects_too_short_input():
    with pytest.raises(ValueError, match="cannot hold"):
        unpack(torch.zeros(1, dtype=torch.uint8), bits=4, count=3)


@pytest.mark.parametrize("bits", [0, 9])
def test_rejects_unsupported_bits(bits):
    with pytest.raises(ValueError, match="bits"):
        pack(torch.zeros(1, dtype=torch.uint8), bits)
