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
def test_cache_on_vs_off_identical_real_fixture(tmp_path: Path) -> None:
    """Tests cache on vs off with real fixture (skipif golden missing)."""
    out_off = tmp_path / "off.tif"
    out_on = tmp_path / "on.tif"
    log = tmp_path / "run.log"

    run_pipeline(
        FIXTURES / "12003.tif", FIXTURES / "12003.txt", out_off, log,
        order=3, exp=-1.0, tile_size=1024, workers=2, shared_cache_mb=0,
    )
    run_pipeline(
        FIXTURES / "12003.tif", FIXTURES / "12003.txt", out_on, log,
        order=3, exp=-1.0, tile_size=1024, workers=2,
        shared_cache_mb=64, cache_key_mode="exact",
    )

    result = compare_outputs(out_off, out_on)
    assert result.is_identical, f"max_diff={result.global_max_diff}"


def test_cache_on_vs_off_identical_synthetic(tmp_path: Path) -> None:
    """Vers\u00e3o sint\u00e9tica que n\u00e3o depende do golden master real — roda sempre."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(3, 24, 24), dtype=np.uint8)
    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)
    samples = tmp_path / "samples.txt"
    samples.write_text("10 20 30\n200 210 220\n128 128 128\n")
    log = tmp_path / "run.log"

    out_off = tmp_path / "off.tif"
    out_on = tmp_path / "on.tif"

    run_pipeline(input_tif, samples, out_off, log,
                order=1, exp=-1.0, tile_size=8, workers=1, shared_cache_mb=0)
    run_pipeline(input_tif, samples, out_on, log,
                order=1, exp=-1.0, tile_size=8, workers=1,
                shared_cache_mb=4, cache_key_mode="exact")

    result = compare_outputs(out_off, out_on)
    assert result.is_identical
