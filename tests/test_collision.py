import numpy as np
from pipeline.cache.shared_direct_mapped import SharedDirectMappedCache
from pipeline.cache.hash_strategy import ConstantHashStrategy


def make_colliding_cache(n_bands: int = 3, dtype: np.dtype = np.uint8) -> "SharedDirectMappedCache":
    return SharedDirectMappedCache(
        size_mb=1, n_bands=n_bands, sig_dtype=dtype,
        hash_strategy=ConstantHashStrategy(),
    )


def test_collision_never_returns_wrong_value() -> None:
    cache = make_colliding_cache()
    sig_a = np.array([[1, 2, 3]], dtype=np.uint8)
    sig_b = np.array([[4, 5, 6]], dtype=np.uint8)
    val_a = np.array([77], dtype=np.uint8)
    cache.insert_many(sig_a, val_a)
    out, hit_mask, stats = cache.lookup_many(sig_b)
    assert not hit_mask[0], "Colisão não deve gerar hit"
    assert stats.n_misses_collision == 1
    cache.close()


def test_collision_does_not_overwrite_original() -> None:
    cache = make_colliding_cache()
    sig_a = np.array([[1, 2, 3]], dtype=np.uint8)
    sig_b = np.array([[4, 5, 6]], dtype=np.uint8)
    cache.insert_many(sig_a, np.array([42], dtype=np.uint8))
    insert_stats = cache.insert_many(sig_b, np.array([99], dtype=np.uint8))
    assert insert_stats.n_skipped_collision == 1
    out, hit_mask, _ = cache.lookup_many(sig_a)
    assert hit_mask[0] and out[0] == 42
    cache.close()


def test_collision_miss_is_recomputed_correctly() -> None:
    cache = make_colliding_cache()
    sig_a = np.array([[10, 20, 30]], dtype=np.uint8)
    sig_b = np.array([[40, 50, 60]], dtype=np.uint8)
    cache.insert_many(sig_a, np.array([111], dtype=np.uint8))
    _, hit_b, _ = cache.lookup_many(sig_b)
    assert not hit_b[0], "B deve ser miss por colisão"
    real_value_b = np.array([222], dtype=np.uint8)
    cache.insert_many(sig_b, real_value_b)
    assert real_value_b[0] == 222
    cache.close()


def test_insert_skipped_collision_counter() -> None:
    cache = make_colliding_cache()
    sig_a = np.array([[1, 2, 3]], dtype=np.uint8)
    sig_b = np.array([[4, 5, 6]], dtype=np.uint8)
    sig_c = np.array([[7, 8, 9]], dtype=np.uint8)
    cache.insert_many(sig_a, np.array([1], dtype=np.uint8))
    both = np.vstack([sig_b, sig_c])
    stats = cache.insert_many(both, np.array([2, 3], dtype=np.uint8))
    assert stats.n_skipped_collision == 2
    cache.close()
