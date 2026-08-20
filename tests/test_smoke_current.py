"""Smoke test: roda main.py atual contra golden master."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden_master"
FIXTURES_DIR = TESTS_DIR / "fixtures"

MAX_ACCEPTABLE_DIFF = 1  # erro de arredondamento float32 vs float64


def test_current_pipeline_matches_golden() -> None:
    input_tif = FIXTURES_DIR / "12003.tif"
    golden_poly = GOLDEN_DIR / "12003-poly.tif"
    samples = FIXTURES_DIR / "12003.txt"

    if not input_tif.exists():
        pytest.skip("Fixture input 12003.tif nao encontrado")
    if not golden_poly.exists():
        pytest.skip("Golden master output 12003-poly.tif nao encontrado")

    from pipeline.compare.outputs import compare_outputs

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        output_path = tmp.name
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.main",
             str(input_tif), output_path, str(samples), "/dev/null",
             "--order", "3", "--exp", "-1.0", "--tile-size", "1024", "--workers", "4"],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"main.py falhou com código {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        comp = compare_outputs(golden_poly, output_path, tile_size=1024)
        assert comp.global_max_diff <= MAX_ACCEPTABLE_DIFF, (
            f"max_diff={comp.global_max_diff} (max={MAX_ACCEPTABLE_DIFF}), "
            f"diff_pixels={comp.total_diff_pixels}/{comp.total_pixels}, "
            f"similarity={comp.global_similarity:.6f}"
        )


def test_single_vs_multi_worker_identical() -> None:
    input_tif = FIXTURES_DIR / "12003.tif"
    samples = FIXTURES_DIR / "12003.txt"

    if not input_tif.exists():
        pytest.skip("Fixture input 12003.tif nao encontrado")

    from pipeline.compare.outputs import compare_outputs

    def run_pipeline(workers: int) -> tempfile.NamedTemporaryFile:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.main",
             str(input_tif), tmp.name, str(samples), "/dev/null",
             "--order", "3", "--exp", "-1.0", "--tile-size", "1024",
             "--workers", str(workers)],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"workers={workers} falhou\n{result.stdout}\n{result.stderr}"
        )
        return tmp

    try:
        tmp1 = run_pipeline(1)
        tmp4 = run_pipeline(4)
        comp = compare_outputs(tmp1.name, tmp4.name, tile_size=1024)
        assert comp.is_identical, (
            f"Single vs multi-worker diverge: max_diff={comp.global_max_diff}"
        )
    finally:
        tmp1.close()
        tmp4.close()
        Path(tmp1.name).unlink(missing_ok=True)
        Path(tmp4.name).unlink(missing_ok=True)


def test_cache_vs_no_cache_identical() -> None:
    """Spec 08 item 2: mesma fixture gera output idêntico com e sem cache."""
    input_tif = FIXTURES_DIR / "12003.tif"
    samples = FIXTURES_DIR / "12003.txt"

    if not input_tif.exists():
        pytest.skip("Fixture input 12003.tif nao encontrado")

    from pipeline.compare.outputs import compare_outputs

    def run_pipeline(shared_cache_mb: int) -> tempfile.NamedTemporaryFile:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.main",
             str(input_tif), tmp.name, str(samples), "/dev/null",
             "--order", "3", "--exp", "-1.0", "--tile-size", "1024",
             "--workers", "4",
             "--shared-cache-mb", str(shared_cache_mb)],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"cache_mb={shared_cache_mb} falhou\n{result.stdout}\n{result.stderr}"
        )
        return tmp

    try:
        tmp_cache = run_pipeline(shared_cache_mb=64)
        tmp_no_cache = run_pipeline(shared_cache_mb=0)
        comp = compare_outputs(tmp_cache.name, tmp_no_cache.name, tile_size=1024)
        assert comp.is_identical, (
            f"Cache vs no-cache diverge: max_diff={comp.global_max_diff}, "
            f"diff={comp.total_diff_pixels}/{comp.total_pixels}"
        )
    finally:
        tmp_cache.close()
        tmp_no_cache.close()
        Path(tmp_cache.name).unlink(missing_ok=True)
        Path(tmp_no_cache.name).unlink(missing_ok=True)
