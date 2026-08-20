"""Unit tests for the audit module — no rasterio or PolyModel dependency."""
import csv
import json
from pathlib import Path

from pipeline.audit.stats import PipelineStats, TileStats
from pipeline.audit.writer import write_audit_json, write_tile_csv


def make_tile_stats(
    tile_id: int = 0,
    valid: int = 100,
    unique: int = 20,
    hits: int = 10,
    misses_e: int = 5,
    misses_c: int = 5,
    inserted: int = 5,
    skipped: int = 5,
) -> TileStats:
    return TileStats(
        tile_id=tile_id,
        row_off=0,
        col_off=0,
        height=32,
        width=32,
        total_pixels=128,
        valid_pixels=valid,
        unique_signatures=unique,
        cache_lookups=unique,
        cache_hits=hits,
        cache_misses_empty=misses_e,
        cache_misses_collision=misses_c,
        cache_inserted=inserted,
        cache_insert_skipped_collision=skipped,
        time_read_tile=0.01,
        time_unique=0.001,
        time_cache_lookup=0.005,
        time_model_eval=0.1,
        time_cache_insert=0.003,
        time_reconstruct=0.002,
        time_write_tile=0.01,
        time_total=0.131,
    )


def test_accumulate_sums_correctly() -> None:
    ps = PipelineStats()
    ps.accumulate(make_tile_stats(tile_id=0, valid=100, unique=20, hits=10))
    ps.accumulate(make_tile_stats(tile_id=1, valid=200, unique=40, hits=20))
    assert ps.total_tiles == 2
    assert ps.valid_pixels == 300
    assert ps.unique_signatures_total == 60
    assert ps.cache_hits == 30


def test_hit_rate_calculation() -> None:
    ps = PipelineStats()
    ts = make_tile_stats(unique=100, hits=75, misses_e=15, misses_c=10)
    ts.cache_lookups = 100
    ps.accumulate(ts)
    assert abs(ps.hit_rate - 0.75) < 1e-6


def test_hit_rate_zero_when_no_lookups() -> None:
    ps = PipelineStats()
    assert ps.hit_rate == 0.0


def test_unique_ratio_min_max() -> None:
    ps = PipelineStats()
    ts1 = make_tile_stats(valid=100, unique=10)
    ts2 = make_tile_stats(valid=100, unique=50)
    ps.accumulate(ts1)
    ps.accumulate(ts2)
    assert abs(ps.unique_ratio_min - 0.1) < 1e-6
    assert abs(ps.unique_ratio_max - 0.5) < 1e-6
    assert abs(ps.unique_ratio_mean - 0.3) < 1e-6


def test_to_dict_has_all_required_fields() -> None:
    required = [
        "cache_enabled",
        "hash_strategy",
        "cache_key_mode",
        "requested_cache_mb",
        "effective_cache_mb",
        "effective_slot_count",
        "cache_lookups",
        "cache_hits",
        "cache_misses_empty",
        "cache_misses_collision",
        "cache_inserted",
        "cache_insert_skipped_collision",
        "hit_rate",
        "collision_miss_rate",
        "time_total",
    ]
    ps = PipelineStats()
    d = ps.to_dict()
    for fld in required:
        assert fld in d, f"Missing in to_dict(): {fld}"


def test_write_audit_json(tmp_path: Path) -> None:
    ps = PipelineStats(cache_enabled=True, hash_strategy="blake2b")
    ps.accumulate(make_tile_stats())
    out = tmp_path / "audit.json"
    write_audit_json(ps, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["cache_enabled"] is True
    assert data["hash_strategy"] == "blake2b"
    assert data["total_tiles"] == 1


def test_write_tile_csv(tmp_path: Path) -> None:
    ps = PipelineStats()
    ps.accumulate(make_tile_stats(tile_id=0))
    ps.accumulate(make_tile_stats(tile_id=1))
    out = tmp_path / "tiles.csv"
    write_tile_csv(ps, str(out))
    assert out.exists()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["tile_id"] == "0"
    assert rows[1]["tile_id"] == "1"


def test_write_tile_csv_required_columns(tmp_path: Path) -> None:
    required_cols = [
        "tile_id",
        "row_off",
        "col_off",
        "height",
        "width",
        "total_pixels",
        "valid_pixels",
        "unique_signatures",
        "unique_ratio",
        "cache_lookups",
        "cache_hits",
        "cache_misses_empty",
        "cache_misses_collision",
        "cache_inserted",
        "cache_insert_skipped_collision",
        "time_read_tile",
        "time_unique",
        "time_cache_lookup",
        "time_model_eval",
        "time_cache_insert",
        "time_reconstruct",
        "time_write_tile",
        "time_total",
    ]
    ps = PipelineStats()
    ps.accumulate(make_tile_stats())
    out = tmp_path / "tiles.csv"
    write_tile_csv(ps, str(out))
    with open(out) as f:
        reader = csv.DictReader(f)
        for col in required_cols:
            assert col in reader.fieldnames, f"Missing CSV column: {col}"


def test_invalid_pixels_property() -> None:
    ts = make_tile_stats(valid=80)
    assert ts.invalid_pixels == 128 - 80
