from pathlib import Path
import numpy as np
import pytest

from tests.helpers import ROOT, run_pipeline, write_synthetic_tif
from pipeline.compare.outputs import compare_outputs

GOLDEN = ROOT / "tests" / "golden_master"
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.mark.skipif(
    not ((FIXTURES / "12003.tif").exists() and (GOLDEN / "12003-poly.tif").exists()),
    reason="Golden master ausente.",
)
def test_single_vs_multi_worker_real_fixture(tmp_path: Path) -> None:
    """Tests single vs multi worker with real fixture (skipif golden missing)."""
    out_1 = tmp_path / "w1.tif"
    out_4 = tmp_path / "w4.tif"
    log = tmp_path / "run.log"

    run_pipeline(FIXTURES / "12003.tif", FIXTURES / "12003.txt", out_1, log,
                order=3, exp=-1.0, tile_size=512, workers=1, shared_cache_mb=64)
    run_pipeline(FIXTURES / "12003.tif", FIXTURES / "12003.txt", out_4, log,
                order=3, exp=-1.0, tile_size=512, workers=4, shared_cache_mb=64)

    result = compare_outputs(out_1, out_4)
    assert result.is_identical, f"max_diff={result.global_max_diff} — poss\u00edvel race condition"


def test_single_vs_multi_worker_synthetic(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    arr = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)
    samples = tmp_path / "samples.txt"
    samples.write_text("10 20 30\n200 210 220\n128 128 128\n50 60 70\n")
    log = tmp_path / "run.log"

    out_1 = tmp_path / "w1.tif"
    out_4 = tmp_path / "w4.tif"

    run_pipeline(input_tif, samples, out_1, log,
                order=2, exp=-1.0, tile_size=16, workers=1, shared_cache_mb=2)
    run_pipeline(input_tif, samples, out_4, log,
                order=2, exp=-1.0, tile_size=16, workers=4, shared_cache_mb=2)

    result = compare_outputs(out_1, out_4)
    assert result.is_identical
