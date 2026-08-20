"""
stats.py — Accumulator de métricas do pipeline.

Thread-safe para acumulação sequencial no processo principal.
Os workers retornam TileStats via multiprocessing; o processo principal
chama accumulate() para cada tile processado.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TileStats:
    """Metrics of a single tile. Returned by the worker for the main process."""
    tile_id: int
    row_off: int
    col_off: int
    height: int
    width: int
    total_pixels: int
    valid_pixels: int

    # np.unique
    unique_signatures: int

    # cache
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses_empty: int = 0
    cache_misses_collision: int = 0
    cache_inserted: int = 0
    cache_insert_skipped_collision: int = 0

    # timing (seconds)
    time_read_tile: float = 0.0
    time_unique: float = 0.0
    time_cache_lookup: float = 0.0
    time_model_eval: float = 0.0
    time_cache_insert: float = 0.0
    time_reconstruct: float = 0.0
    time_write_tile: float = 0.0
    time_total: float = 0.0

    @property
    def invalid_pixels(self) -> int:
        return self.total_pixels - self.valid_pixels

    @property
    def unique_ratio(self) -> float:
        if self.valid_pixels == 0:
            return 0.0
        return self.unique_signatures / self.valid_pixels


@dataclass
class PipelineStats:
    """Global accumulated metrics from all tiles."""
    # Config
    n_bands: int = 0
    input_dtype: str = ""
    output_dtype: str = "uint8"
    tile_size: int = 0
    workers: int = 0

    # Cache config
    cache_enabled: bool = False
    cache_backend: str = "NullCache"
    hash_strategy: str = ""
    cache_key_mode: str = "exact"
    cache_key_decimals: int | None = None
    requested_cache_mb: float = 0.0
    effective_cache_mb: float = 0.0
    effective_cache_bytes: int = 0
    slot_size_bytes: int = 0
    effective_slot_count: int = 0

    # Totals
    total_tiles: int = 0
    total_pixels: int = 0
    valid_pixels: int = 0
    invalid_pixels: int = 0

    # Cache totals
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses_empty: int = 0
    cache_misses_collision: int = 0
    cache_insert_attempts: int = 0
    cache_inserted: int = 0
    cache_insert_skipped_collision: int = 0
    cache_occupancy_estimated: float = 0.0

    # Unique signatures
    unique_signatures_total: int = 0

    # Timing totals (seconds)
    time_read_tile: float = 0.0
    time_unique: float = 0.0
    time_cache_lookup: float = 0.0
    time_model_eval: float = 0.0
    time_cache_insert: float = 0.0
    time_reconstruct: float = 0.0
    time_write_tile: float = 0.0
    time_total: float = 0.0

    # Per-tile raw (for min/max of unique_ratio)
    _tile_unique_ratios: list[float] = field(default_factory=list, repr=False)
    _tile_stats: list[TileStats] = field(default_factory=list, repr=False)

    def accumulate(self, ts: TileStats) -> None:
        """Accumulate metrics from a tile into the global state."""
        self.total_tiles += 1
        self.total_pixels += ts.total_pixels
        self.valid_pixels += ts.valid_pixels
        self.invalid_pixels += ts.invalid_pixels

        self.cache_lookups += ts.cache_lookups
        self.cache_hits += ts.cache_hits
        self.cache_misses_empty += ts.cache_misses_empty
        self.cache_misses_collision += ts.cache_misses_collision
        self.cache_insert_attempts += ts.cache_lookups  # attempts = lookups
        self.cache_inserted += ts.cache_inserted
        self.cache_insert_skipped_collision += ts.cache_insert_skipped_collision

        self.unique_signatures_total += ts.unique_signatures

        self.time_read_tile += ts.time_read_tile
        self.time_unique += ts.time_unique
        self.time_cache_lookup += ts.time_cache_lookup
        self.time_model_eval += ts.time_model_eval
        self.time_cache_insert += ts.time_cache_insert
        self.time_reconstruct += ts.time_reconstruct
        self.time_write_tile += ts.time_write_tile
        self.time_total += ts.time_total

        self._tile_unique_ratios.append(ts.unique_ratio)
        self._tile_stats.append(ts)

    @property
    def hit_rate(self) -> float:
        if self.cache_lookups == 0:
            return 0.0
        return self.cache_hits / self.cache_lookups

    @property
    def collision_miss_rate(self) -> float:
        if self.cache_lookups == 0:
            return 0.0
        return self.cache_misses_collision / self.cache_lookups

    @property
    def unique_ratio_mean(self) -> float:
        if not self._tile_unique_ratios:
            return 0.0
        return sum(self._tile_unique_ratios) / len(self._tile_unique_ratios)

    @property
    def unique_ratio_min(self) -> float:
        return min(self._tile_unique_ratios) if self._tile_unique_ratios else 0.0

    @property
    def unique_ratio_max(self) -> float:
        return max(self._tile_unique_ratios) if self._tile_unique_ratios else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (without private fields)."""
        return {
            "n_bands": self.n_bands,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "tile_size": self.tile_size,
            "workers": self.workers,
            "cache_enabled": self.cache_enabled,
            "cache_backend": self.cache_backend,
            "hash_strategy": self.hash_strategy,
            "cache_key_mode": self.cache_key_mode,
            "cache_key_decimals": self.cache_key_decimals,
            "requested_cache_mb": self.requested_cache_mb,
            "effective_cache_mb": self.effective_cache_mb,
            "effective_cache_bytes": self.effective_cache_bytes,
            "slot_size_bytes": self.slot_size_bytes,
            "effective_slot_count": self.effective_slot_count,
            "total_tiles": self.total_tiles,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "invalid_pixels": self.invalid_pixels,
            "cache_lookups": self.cache_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses_empty": self.cache_misses_empty,
            "cache_misses_collision": self.cache_misses_collision,
            "cache_insert_attempts": self.cache_insert_attempts,
            "cache_inserted": self.cache_inserted,
            "cache_insert_skipped_collision": self.cache_insert_skipped_collision,
            "cache_occupancy_estimated": self.cache_occupancy_estimated,
            "hit_rate": self.hit_rate,
            "collision_miss_rate": self.collision_miss_rate,
            "unique_signatures_total": self.unique_signatures_total,
            "unique_ratio_mean": self.unique_ratio_mean,
            "unique_ratio_min": self.unique_ratio_min,
            "unique_ratio_max": self.unique_ratio_max,
            "time_read_tile": self.time_read_tile,
            "time_unique": self.time_unique,
            "time_cache_lookup": self.time_cache_lookup,
            "time_model_eval": self.time_model_eval,
            "time_cache_insert": self.time_cache_insert,
            "time_reconstruct": self.time_reconstruct,
            "time_write_tile": self.time_write_tile,
            "time_total": self.time_total,
        }
