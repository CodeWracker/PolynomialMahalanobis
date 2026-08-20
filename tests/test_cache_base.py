import numpy as np
import pytest
from src.cache.base import BaseCache
from src.cache.null_cache import NullCache
from src.cache.shared_direct_mapped import SharedDirectMappedCache
from src.cache.hash_strategy import Blake2bHashStrategy, ConstantHashStrategy


# --- NullCache tests ---

def test_null_cache_always_miss() -> None:
    cache = NullCache()
    keys = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    values_out, hit_mask, stats = cache.lookup_many(keys)
    assert not cache.enabled
    assert not np.any(hit_mask)
    assert stats.n_hits == 0
    assert stats.n_misses_empty == 2


def test_null_cache_insert_is_noop() -> None:
    cache = NullCache()
    keys = np.array([[1, 2, 3]], dtype=np.uint8)
    values = np.array([42], dtype=np.uint8)
    insert_stats = cache.insert_many(keys, values)
    assert insert_stats.n_inserted == 0
    _, hit_mask, _ = cache.lookup_many(keys)
    assert not np.any(hit_mask)


# --- BaseCache instantiation test ---

def test_base_cache_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseCache()


# --- SharedDirectMappedCache tests ---

def make_cache(size_mb: float = 1, n_bands: int = 3, dtype: np.dtype = np.uint8, strategy: object | None = None) -> "SharedDirectMappedCache":
    return SharedDirectMappedCache(
        size_mb=size_mb, n_bands=n_bands, sig_dtype=dtype,
        hash_strategy=strategy or Blake2bHashStrategy(),
    )


def test_basic_insert_and_lookup() -> None:
    cache = make_cache()
    keys = np.array([[79, 137, 203]], dtype=np.uint8)
    values = np.array([42], dtype=np.uint8)
    cache.insert_many(keys, values)
    out, hit_mask, stats = cache.lookup_many(keys)
    assert hit_mask[0]
    assert out[0] == 42
    assert stats.n_hits == 1
    cache.close()


def test_lookup_on_empty_cache_is_miss() -> None:
    cache = make_cache()
    keys = np.array([[1, 2, 3]], dtype=np.uint8)
    _, hit_mask, stats = cache.lookup_many(keys)
    assert not hit_mask[0]
    assert stats.n_misses_empty == 1
    cache.close()


def test_different_signatures_independent() -> None:
    cache = make_cache()
    k1 = np.array([[10, 20, 30]], dtype=np.uint8)
    k2 = np.array([[40, 50, 60]], dtype=np.uint8)
    cache.insert_many(k1, np.array([100], dtype=np.uint8))
    cache.insert_many(k2, np.array([200], dtype=np.uint8))
    v1, h1, _ = cache.lookup_many(k1)
    v2, h2, _ = cache.lookup_many(k2)
    assert h1[0] and v1[0] == 100
    assert h2[0] and v2[0] == 200
    cache.close()


def test_insert_same_signature_twice_does_not_overwrite() -> None:
    cache = make_cache()
    keys = np.array([[1, 2, 3]], dtype=np.uint8)
    insert_1 = cache.insert_many(keys, np.array([10], dtype=np.uint8))
    insert_2 = cache.insert_many(keys, np.array([99], dtype=np.uint8))
    out, hit_mask, _ = cache.lookup_many(keys)
    assert hit_mask[0]
    assert out[0] == 10  # first value preserved
    assert insert_1.n_inserted == 1
    assert insert_2.n_skipped_collision == 0
    cache.close()


def test_enabled_is_true() -> None:
    cache = make_cache()
    assert cache.enabled
    cache.close()


def test_cache_info_has_required_fields() -> None:
    cache = make_cache()
    info = cache.info()
    for field in ["cache_backend", "slot_size_bytes", "effective_slot_count",
                  "effective_cache_bytes", "effective_cache_mb", "n_bands",
                  "sig_dtype", "n_locks"]:
        assert field in info, f"Campo ausente: {field}"
    cache.close()
