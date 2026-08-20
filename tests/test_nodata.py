import json
from pathlib import Path

import numpy as np
import rasterio

from pipeline.compare.outputs import compare_outputs
from tests.helpers import run_pipeline, write_synthetic_tif


def test_nodata_pixels_excluded_and_output_zero(tmp_path: Path) -> None:
    arr = np.array([
        [[10, 10], [200, 200]],
        [[10, 10], [200, 200]],
        [[10, 10], [200, 200]],
    ], dtype=np.uint8)
    mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)  # pixel (0,1) inv\u00e1lido

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr, mask=mask)
    samples = tmp_path / "samples.txt"
    samples.write_text("10 10 10\n200 200 200\n")
    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"
    audit = tmp_path / "audit.json"

    run_pipeline(input_tif, samples, out, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0,
                audit_json=str(audit))

    with rasterio.open(out) as ds:
        data = ds.read(1)
    assert data[0, 1] == 0, "Pixel marcado como inv\u00e1lido deve ter sa\u00edda 0"

    audit_data = json.loads(audit.read_text())
    assert audit_data["valid_pixels"] == 3
    assert audit_data["invalid_pixels"] == 1
    assert audit_data["total_pixels"] == 4


def test_nodata_does_not_affect_valid_pixel_computation(tmp_path: Path) -> None:
    """
    Pixels v\u00e1lidos devem ter o mesmo resultado COM ou SEM o pixel inv\u00e1lido presente
    no tile (prova que o inv\u00e1lido n\u00e3o contamina np.unique/cache/modelo).
    """

    arr_with_invalid = np.array([
        [[10, 10], [200, 200]],
        [[10, 10], [200, 200]],
        [[10, 10], [200, 200]],
    ], dtype=np.uint8)
    mask_with_invalid = np.array([[255, 0], [255, 255]], dtype=np.uint8)

    arr_all_valid = arr_with_invalid.copy()
    mask_all_valid = np.full((2, 2), 255, dtype=np.uint8)

    samples_path = tmp_path / "samples.txt"
    samples_path.write_text("10 10 10\n200 200 200\n")

    in_a = tmp_path / "a.tif"
    in_b = tmp_path / "b.tif"
    write_synthetic_tif(in_a, arr_with_invalid, mask=mask_with_invalid)
    write_synthetic_tif(in_b, arr_all_valid, mask=mask_all_valid)

    out_a = tmp_path / "out_a.tif"
    out_b = tmp_path / "out_b.tif"
    log = tmp_path / "run.log"

    run_pipeline(in_a, samples_path, out_a, log, order=1, exp=-1.0,
                tile_size=4, workers=1, shared_cache_mb=0)
    run_pipeline(in_b, samples_path, out_b, log, order=1, exp=-1.0,
                tile_size=4, workers=1, shared_cache_mb=0)

    with rasterio.open(out_a) as da, rasterio.open(out_b) as db:
        a = da.read(1)
        b = db.read(1)
    # pixel (0,0) e linha 1 s\u00e3o v\u00e1lidos em ambos e devem ter resultado ID\u00caNTICO
    assert a[0, 0] == b[0, 0]
    assert (a[1, :] == b[1, :]).all()


def test_all_nodata_image_produces_all_zeros(tmp_path: Path) -> None:
    """Every pixel is nodata (mask=0) — output should be all zeros."""
    arr = np.array([
        [[50, 50], [60, 60]],
        [[50, 50], [60, 60]],
        [[50, 50], [60, 60]],
    ], dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=np.uint8)  # all invalid

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr, mask=mask)
    samples = tmp_path / "samples.txt"
    samples.write_text("50 50 50\n60 60 60\n")
    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"

    run_pipeline(input_tif, samples, out, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0)

    with rasterio.open(out) as ds:
        data = ds.read(1)
    assert data.shape == (2, 2)
    assert (data == 0).all(), "All nodata image must produce all zeros"


def test_single_pixel_image(tmp_path: Path) -> None:
    """1x1 image with valid pixel — should process correctly."""
    arr = np.array([
        [[42]],
        [[42]],
        [[42]],
    ], dtype=np.uint8)
    mask = np.array([[255]], dtype=np.uint8)

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr, mask=mask)
    samples = tmp_path / "samples.txt"
    samples.write_text("42 42 42\n10 10 10\n")
    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"

    run_pipeline(input_tif, samples, out, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0)

    with rasterio.open(out) as ds:
        data = ds.read(1)
    assert data.shape == (1, 1)
    assert isinstance(data[0, 0], (int, np.integer)), "Single pixel should produce a finite value"


def test_single_pixel_all_nodata(tmp_path: Path) -> None:
    """1x1 image with nodata — output should be 0."""
    arr = np.array([
        [[99]],
        [[99]],
        [[99]],
    ], dtype=np.uint8)
    mask = np.array([[0]], dtype=np.uint8)

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr, mask=mask)
    samples = tmp_path / "samples.txt"
    samples.write_text("99 99 99\n10 10 10\n")
    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"

    run_pipeline(input_tif, samples, out, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0)

    with rasterio.open(out) as ds:
        data = ds.read(1)
    assert data.shape == (1, 1)
    assert data[0, 0] == 0, "Single nodata pixel should output 0"
