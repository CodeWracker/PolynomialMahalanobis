import numpy as np
import pytest
from pipeline.main import build_cache


def test_zero_mb_returns_null_cache() -> None:
    cache = build_cache(0.0, n_bands=3, sig_dtype=np.uint8, hash_strategy_name="blake2b")
    assert type(cache).__name__ == "NullCache"


def test_positive_mb_returns_shared_cache() -> None:
    cache = build_cache(1.0, n_bands=3, sig_dtype=np.uint8, hash_strategy_name="blake2b")
    assert type(cache).__name__ == "SharedDirectMappedCache"
    cache.close()


def test_xxhash_strategy_selectable() -> None:
    cache = build_cache(1.0, n_bands=3, sig_dtype=np.uint8, hash_strategy_name="xxhash")
    assert type(cache).__name__ == "SharedDirectMappedCache"
    cache.close()
