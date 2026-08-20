from pathlib import Path
import csv
import json

import numpy as np

from tests.helpers import run_pipeline, write_synthetic_tif

REQUIRED_JSON_FIELDS = [
    "cache_enabled", "hash_strategy", "cache_key_mode", "requested_cache_mb",
    "effective_cache_mb", "effective_slot_count", "cache_lookups", "cache_hits",
    "cache_misses_empty", "cache_misses_collision", "cache_inserted",
    "cache_insert_skipped_collision", "hit_rate", "collision_miss_rate", "time_total",
]


def test_audit_json_has_required_fields(tmp_path: Path) -> None:
    arr = np.random.default_rng(3).integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)
    samples = tmp_path / "samples.txt"
    samples.write_text("10 20 30\n200 210 220\n")
    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"
    audit = tmp_path / "audit.json"

    run_pipeline(input_tif, samples, out, log,
                order=1, exp=-1.0, tile_size=8, workers=1, shared_cache_mb=1,
                audit_json=str(audit))

    data = json.loads(audit.read_text())
    for field in REQUIRED_JSON_FIELDS:
        assert field in data, f"Campo obrigat\u00f3rio ausente no audit JSON: {field}"


def test_audit_tile_csv_one_row_per_tile(tmp_path: Path) -> None:
    arr = np.random.default_rng(5).integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)
    samples = tmp_path / "samples.txt"
    samples.write_text("10 20 30\n200 210 220\n")
    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"
    tile_csv = tmp_path / "tiles.csv"

    # tile_size=4 numa imagem 8x8 -> 4 tiles
    run_pipeline(input_tif, samples, out, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0,
                audit_tile_csv=str(tile_csv))

    rows = list(csv.DictReader(open(tile_csv)))  # noqa: SIM115
    assert len(rows) == 4, f"Esperado 4 tiles, encontrado {len(rows)}"
