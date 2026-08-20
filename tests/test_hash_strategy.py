import pytest
from pipeline.cache.hash_strategy import Blake2bHashStrategy, XxHashStrategy, ConstantHashStrategy


def test_blake2b_is_deterministic() -> None:
    h = Blake2bHashStrategy()
    data = b"\x01\x02\x03"
    assert h.hash(data) == h.hash(data)


def test_blake2b_different_data_different_hash() -> None:
    h = Blake2bHashStrategy()
    assert h.hash(b"\x01") != h.hash(b"\x02")


def test_blake2b_returns_nonnegative() -> None:
    h = Blake2bHashStrategy()
    assert h.hash(b"test") >= 0


def test_xxhash_is_deterministic() -> None:
    h = XxHashStrategy()
    data = b"\xAA\xBB\xCC"
    assert h.hash(data) == h.hash(data)


def test_xxhash_different_data_different_hash() -> None:
    h = XxHashStrategy()
    assert h.hash(b"\x01") != h.hash(b"\x02")


def test_constant_always_returns_zero() -> None:
    h = ConstantHashStrategy()
    assert h.hash(b"abc") == 0
    assert h.hash(b"xyz") == 0
    assert h.hash(b"") == 0


def test_all_strategies_same_interface() -> None:
    for Cls in [Blake2bHashStrategy, XxHashStrategy, ConstantHashStrategy]:
        h = Cls()
        result = h.hash(b"\x01\x02\x03")
        assert isinstance(result, int)
        assert result >= 0
