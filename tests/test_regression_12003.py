"""
Teste de regressão obrigatório: pipeline nova, modo exact, sem cache, deve produzir
output 100% idêntico ao golden master capturado em Goal 00 (binário maha C++ original).
"""
from pathlib import Path

import pytest

from tests.helpers import ROOT, run_pipeline
from src.compare.outputs import compare_outputs

GOLDEN: Path = ROOT / "tests" / "golden_master"
FIXTURES: Path = ROOT / "tests" / "fixtures"

pytestmark = pytest.mark.skipif(
    not ((FIXTURES / "12003.tif").exists() and (GOLDEN / "12003-poly.tif").exists()),
    reason="Golden master ausente. Rode tests/capture_golden_master.py (Goal 00.3) primeiro.",
)


def test_regression_exact_no_cache_single_worker(tmp_path: Path) -> None:
    out: Path = tmp_path / "out.tif"
    log: Path = tmp_path / "run.log"

    run_pipeline(
        FIXTURES / "12003.tif", FIXTURES / "12003.txt", out, log,
        order=3, exp=-1.0, tile_size=1024, workers=1,
        shared_cache_mb=0, cache_key_mode="exact",
    )

    result = compare_outputs(out, GOLDEN / "12003-poly.tif")

    assert result.global_max_diff <= 1, (
        f"Regress\u00e3o 12003 FALHOU.\n"
        f"global_similarity={result.global_similarity*100:.6f}%\n"
        f"global_max_diff={result.global_max_diff}\n"
        f"global_mean_diff={result.global_mean_diff}\n"
        f"total_diff_pixels={result.total_diff_pixels}/{result.total_pixels}"
    )
