import numpy as np
import pytest
from src.cache.shared_direct_mapped import SharedDirectMappedCache
from src.cache.hash_strategy import Blake2bHashStrategy, ConstantHashStrategy


def make_tiny_cache(n_bands: int = 3, dtype: np.dtype | np.integer | type = np.uint8) -> SharedDirectMappedCache:
    return SharedDirectMappedCache(
        size_mb=0.00005, n_bands=n_bands, sig_dtype=dtype,
        hash_strategy=Blake2bHashStrategy(),
    )


def test_full_cache_does_not_crash() -> None:
    cache = make_tiny_cache()
    assert cache.n_slots <= 20, f"Cache deveria ter <= 20 slots, tem {cache.n_slots}"
    rng = np.random.default_rng(42)
    sigs = rng.integers(0, 256, size=(100, 3), dtype=np.uint8)
    vals = rng.integers(0, 256, size=100, dtype=np.uint8)
    insert_stats = cache.insert_many(sigs, vals)
    assert insert_stats.n_attempts == 100
    assert insert_stats.n_inserted + insert_stats.n_skipped_collision == 100
    _, _, lookup_stats = cache.lookup_many(sigs)
    assert lookup_stats.n_queries == 100
    cache.close()


def test_full_cache_result_equal_to_no_cache() -> None:
    cache = make_tiny_cache()
    rng = np.random.default_rng(123)
    sigs = rng.integers(0, 256, size=(50, 3), dtype=np.uint8)

    def fake_model(sig: np.ndarray) -> int:
        return int(sig.sum()) % 256

    expected = np.array([fake_model(s) for s in sigs], dtype=np.uint8)
    out_values, hit_mask, _ = cache.lookup_many(sigs)
    result = np.empty(len(sigs), dtype=np.uint8)
    result[hit_mask] = out_values[hit_mask]
    miss_sigs = sigs[~hit_mask]
    miss_vals = np.array([fake_model(s) for s in miss_sigs], dtype=np.uint8)
    result[~hit_mask] = miss_vals
    cache.insert_many(miss_sigs, miss_vals)
    np.testing.assert_array_equal(result, expected,
        err_msg="Resultado com cache cheio deve ser idêntico ao sem cache")
    cache.close()


def test_write_on_empty_does_not_overwrite_occupied_slot() -> None:
    cache = SharedDirectMappedCache(
        size_mb=1, n_bands=3, sig_dtype=np.uint8,
        hash_strategy=ConstantHashStrategy(),
    )
    sig_a = np.array([[1, 1, 1]], dtype=np.uint8)
    sig_b = np.array([[2, 2, 2]], dtype=np.uint8)
    cache.insert_many(sig_a, np.array([10], dtype=np.uint8))
    stats = cache.insert_many(sig_b, np.array([20], dtype=np.uint8))
    assert stats.n_skipped_collision == 1
    out, hit, _ = cache.lookup_many(sig_a)
    assert hit[0] and out[0] == 10
    cache.close()
