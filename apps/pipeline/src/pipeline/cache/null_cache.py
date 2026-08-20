"""
null_cache.py — Implementação de cache nulo (disabled).

Usado quando --shared-cache-mb 0.
Sempre retorna miss, nunca insere nada.
Garante que a pipeline funcione identicamente com e sem cache (em modo exact).
"""
import numpy as np
from .base import BaseCache, LookupStats, InsertStats


class NullCache(BaseCache):
    """Cache desabilitado. Sempre miss, insert é no-op."""

    @property
    def enabled(self) -> bool:
        return False

    def lookup_many(
        self, key_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, LookupStats]:
        n = len(key_array)
        values = np.zeros(n, dtype=np.uint8)
        hit_mask = np.zeros(n, dtype=bool)
        stats = LookupStats(
            n_queries=n,
            n_hits=0,
            n_misses_empty=n,
            n_misses_collision=0,
        )
        return values, hit_mask, stats

    def insert_many(
        self, key_array: np.ndarray, values: np.ndarray
    ) -> InsertStats:
        return InsertStats(
            n_attempts=len(key_array),
            n_inserted=0,
            n_skipped_collision=0,
        )
