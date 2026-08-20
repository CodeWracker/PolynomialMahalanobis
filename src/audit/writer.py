"""
writer.py — JSON and CSV writers for audit output.
"""
import csv
import json
from pathlib import Path


from .stats import PipelineStats


def write_audit_json(stats: PipelineStats, path: str | Path) -> None:
    """Write global audit JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)


def write_tile_csv(stats: PipelineStats, path: str | Path) -> None:
    """
    Write CSV with one row per tile.
    Only writes if stats._tile_stats is not empty.
    """
    if not stats._tile_stats:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    FIELDS: list[str] = [
        "tile_id", "row_off", "col_off", "height", "width",
        "total_pixels", "valid_pixels", "unique_signatures", "unique_ratio",
        "cache_lookups", "cache_hits", "cache_misses_empty", "cache_misses_collision",
        "cache_inserted", "cache_insert_skipped_collision",
        "time_read_tile", "time_unique", "time_cache_lookup", "time_model_eval",
        "time_cache_insert", "time_reconstruct", "time_write_tile", "time_total",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for ts in stats._tile_stats:
            writer.writerow({
                "tile_id": ts.tile_id,
                "row_off": ts.row_off,
                "col_off": ts.col_off,
                "height": ts.height,
                "width": ts.width,
                "total_pixels": ts.total_pixels,
                "valid_pixels": ts.valid_pixels,
                "unique_signatures": ts.unique_signatures,
                "unique_ratio": round(ts.unique_ratio, 6),
                "cache_lookups": ts.cache_lookups,
                "cache_hits": ts.cache_hits,
                "cache_misses_empty": ts.cache_misses_empty,
                "cache_misses_collision": ts.cache_misses_collision,
                "cache_inserted": ts.cache_inserted,
                "cache_insert_skipped_collision": ts.cache_insert_skipped_collision,
                "time_read_tile": round(ts.time_read_tile, 6),
                "time_unique": round(ts.time_unique, 6),
                "time_cache_lookup": round(ts.time_cache_lookup, 6),
                "time_model_eval": round(ts.time_model_eval, 6),
                "time_cache_insert": round(ts.time_cache_insert, 6),
                "time_reconstruct": round(ts.time_reconstruct, 6),
                "time_write_tile": round(ts.time_write_tile, 6),
                "time_total": round(ts.time_total, 6),
            })
