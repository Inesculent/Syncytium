from itertools import combinations

import pytest
import torch

from syncytium.shards.erasure import decode, encode, generator, inverse, multiply


def random_bytes(shape, seed=0):
    return torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(seed)).to(
        torch.uint8
    )


def test_every_nonzero_element_has_an_inverse():
    assert all(multiply(a, inverse(a)) == 1 for a in range(1, 256))
    with pytest.raises(ZeroDivisionError):
        inverse(0)


def test_multiplication_distributes_over_xor():
    for a, b, c in [(3, 7, 200), (255, 128, 1), (29, 0, 91), (142, 142, 17)]:
        assert multiply(a, b ^ c) == multiply(a, b) ^ multiply(a, c)


def test_multiplication_matches_carryless_product_reduced():
    def slow(a, b):
        product = 0
        while b:
            if b & 1:
                product ^= a
            a, b = a << 1, b >> 1
            if a & 0x100:
                a ^= 0x11D
        return product

    assert all(multiply(a, b) == slow(a, b) for a in range(0, 256, 7) for b in range(256))


@pytest.mark.parametrize(("num_shares", "num_data"), [(2, 1), (4, 3), (4, 2), (5, 5), (8, 4)])
def test_any_k_shares_recover_the_data(num_shares, num_data):
    data = random_bytes((num_data, 33))
    shares = encode(data, num_shares)
    assert torch.equal(shares[:num_data], data)  # systematic
    for chosen in combinations(range(num_shares), num_data):
        assert torch.equal(decode({i: shares[i] for i in chosen}, num_data, num_shares), data)


def test_largest_code_recovers_from_random_subsets():
    data = random_bytes((128, 5))
    shares = encode(data, 255)
    for seed in range(3):
        chosen = torch.randperm(255, generator=torch.Generator().manual_seed(seed))[:128]
        available = {int(i): shares[i] for i in chosen}
        assert torch.equal(decode(available, 128, 255), data)


def test_too_few_shares_are_rejected():
    shares = encode(random_bytes((3, 4)), 5)
    with pytest.raises(ValueError, match="need 3 of the 5"):
        decode({0: shares[0], 4: shares[4]}, 3, 5)


@pytest.mark.parametrize(("num_shares", "num_data"), [(256, 2), (3, 4), (3, 0)])
def test_impossible_codes_are_rejected(num_shares, num_data):
    with pytest.raises(ValueError, match="k ≤ n"):
        generator(num_shares, num_data)
