import json
from pathlib import Path

import numpy as np

from src.compare.outputs import compare_outputs
from tests.helpers import run_pipeline, write_synthetic_tif


def test_exact_mode_does_not_collapse_close_float32(tmp_path: Path) -> None:
    """
    Test exact mode preserves distinct float32 values.

    2-band image, 1 pixel per unique signature.
    Expected 2 unique signatures (pixels differ in float32 bytes).
    """
    # (n_bands=2, height=1, width=2) -> 2 pixels with different float32 values
    # pixel0: [0.123457, 0.523457], pixel1: [0.123508, 0.523508]
    arr = np.array(
        [[[0.123457, 0.123508]], [[0.523457, 0.523508]]],
        dtype=np.float32,
    )
    # shape (2, 1, 2) -> 2 bands, 1 row, 2 cols

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)

    samples = tmp_path / "samples.txt"
    samples.write_text(
        "0.100 0.500\n"
        "0.900 0.900\n"
        "0.500 0.500\n"
    )

    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"
    audit = tmp_path / "audit.json"

    run_pipeline(
        input_tif, samples, out, log,
        order=1, exp=-1.0, tile_size=16, workers=1, shared_cache_mb=0,
        cache_key_mode="exact", audit_json=str(audit),
    )

    data = json.loads(audit.read_text())
    assert data["unique_signatures_total"] == 2, (
        f"Modo exact deve preservar values float32 distintas. "
        f"Got {data['unique_signatures_total']}."
    )


def test_round_mode_can_collapse_close_float32(tmp_path: Path) -> None:
    """
    Test round mode collapses close float32 values.

    2-band image, 2 pixels near each other in float32 space.
    Round mode with 4 decimals should collapse to 1 signature.
    """
    # Two pixels that differ in float32 but round to same value at 4 decimals
    # pixel0: [0.123457, 0.523457] -> rounds to [0.1235, 0.5235]
    # pixel1: [0.123499, 0.523499] -> rounds to [0.1235, 0.5235]
    arr = np.array(
        [[[0.123457, 0.123499]], [[0.523457, 0.523499]]],
        dtype=np.float32,
    )
    # shape (2, 1, 2)

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)

    samples = tmp_path / "samples.txt"
    samples.write_text(
        "0.100 0.500\n"
        "0.900 0.900\n"
        "0.500 0.500\n"
    )

    out = tmp_path / "out.tif"
    log = tmp_path / "run.log"
    audit = tmp_path / "audit.json"

    run_pipeline(
        input_tif, samples, out, log,
        order=1, exp=-1.0, tile_size=16, workers=1, shared_cache_mb=0,
        cache_key_mode="round", cache_key_decimals=4, audit_json=str(audit),
    )

    data = json.loads(audit.read_text())
    assert data["unique_signatures_total"] == 1, (
        f"Modo round com 4 casas deve colapsar valores prximos. "
        f"Got {data['unique_signatures_total']}."
    )


def test_round_mode_report_does_not_assert_equality_to_exact(tmp_path: Path) -> None:
    """
    Documents POLICY: round mode evaluates via similarity report,
    not mandatory equality. Runs exact and round, compares,
    verifies both succeed without requiring 100% match.
    """
    # 3-band float32 image, 1 row x 2 cols
    arr = np.array(
        [[[0.10001, 0.20001]], [[0.30001, 0.40001]], [[0.50001, 0.60001]]],
        dtype=np.float32,
    )
    # shape (3, 1, 2)

    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)

    samples = tmp_path / "samples.txt"
    samples.write_text(
        "0.1 0.3 0.5\n"
        "0.9 0.9 0.9\n"
        "0.2 0.4 0.6\n"
    )

    out_exact = tmp_path / "exact.tif"
    out_round = tmp_path / "round.tif"
    log = tmp_path / "run.log"

    run_pipeline(
        input_tif, samples, out_exact, log,
        order=1, exp=-1.0, tile_size=16, workers=1, shared_cache_mb=0,
        cache_key_mode="exact",
    )
    run_pipeline(
        input_tif, samples, out_round, log,
        order=1, exp=-1.0, tile_size=16, workers=1, shared_cache_mb=0,
        cache_key_mode="round", cache_key_decimals=2,
    )

    result = compare_outputs(out_exact, out_round)
    assert result.total_pixels > 0
