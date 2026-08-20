"""Tests for benchmark_full.py."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

import benchmark_full


def _write_tif(path: Path, arr: np.ndarray) -> None:
    h, w = arr.shape
    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        str(path), "w",
        driver="GTiff", dtype="uint8",
        count=1, height=h, width=w,
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(arr.astype(np.uint8), 1)


class TestBuildMatrix:
    def test_total_configurations(self) -> None:
        matrix = benchmark_full._build_matrix()
        # 6 cache sizes x 3 workers = 18
        # but cache 0 only has 1 strategy entry, others have 2
        # cache=0: 3 configs (no hash)
        # cache>0: 5 sizes x 2 strategies x 3 workers = 30
        # total = 33
        assert len(matrix) == 183

    def test_cache_zero_no_strategy(self) -> None:
        matrix = benchmark_full._build_matrix()
        zero_cache = [c for c in matrix if c["shared_cache_mb"] == 0]
        assert len(zero_cache) == 3  # workers 1,2,4
        for c in zero_cache:
            assert c["hash_strategy"] == ""

    def test_all_workers_present(self) -> None:
        matrix = benchmark_full._build_matrix()
        for cache_mb in [0, 16, 64, 128, 256, 512]:
            subset = [c for c in matrix if c["shared_cache_mb"] == cache_mb]
            workers = {c["workers"] for c in subset}
            assert {1, 2, 4} <= workers

    def test_strategies_for_cached(self) -> None:
        matrix = benchmark_full._build_matrix()
        for cache_mb in [16, 64, 128, 256, 512]:
            subset = [c for c in matrix if c["shared_cache_mb"] == cache_mb]
            strats = {c["hash_strategy"] for c in subset}
            assert {"blake2b", "xxhash"} <= strats

class TestConfigDesc:
    def test_no_cache_desc(self) -> None:
        cfg = {"shared_cache_mb": 0, "hash_strategy": "", "workers": 2, "cache_key_mode": "exact"}
        assert benchmark_full._config_desc(cfg) == "no-cache_w2_exact"

    def test_cached_desc(self) -> None:
        cfg = {"shared_cache_mb": 128, "hash_strategy": "xxhash", "workers": 4, "cache_key_mode": "exact", "cache_key_decimals": None}
        assert benchmark_full._config_desc(cfg) == "cache128MB_xxhash_w4_exact"

class TestFormatTerminalTable:
    def test_empty_results(self) -> None:
        assert benchmark_full._format_terminal_table([]) == "No results."

    def test_single_result(self) -> None:
        r = benchmark_full.BenchResult(
            config_desc="no-cache_w1", shared_cache_mb=0, hash_strategy="",
            cache_key_mode="exact", cache_key_decimals=0,
            workers=1, wall_clock_s=50.0, hit_rate=0.0, miss_empty=0,
            miss_collision=0, inserted=0, similarity=1.0, diff_pct=0.0,
            max_pixel_diff=0, mean_diff=0.0
        )
        table = benchmark_full._format_terminal_table([r])
        assert "no-cache_w1" in table
        assert "50.00" in table
        assert "1.000000" in table

    def test_multiple_results_format(self) -> None:
        results = []
        for w in [1, 2, 4]:
            results.append(benchmark_full.BenchResult(
                config_desc=f"no-cache_w{w}", shared_cache_mb=0, hash_strategy="",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=w, wall_clock_s=100.0 / w, hit_rate=0.0, miss_empty=0,
                miss_collision=0, inserted=0, similarity=1.0, diff_pct=0.0,
                max_pixel_diff=0, mean_diff=0.0
            ))
        table = benchmark_full._format_terminal_table(results)
        lines = table.split(benchmark_full.NL)
        assert len(lines) >= 4  # header + sep + 3 data
        assert "|" in lines[0]
        assert "-+-" in lines[1]


class TestWriteCsv:
    def test_csv_structure(self, tmp_path: Path) -> None:
        results = [
            benchmark_full.BenchResult(
                config_desc="no-cache_w1", shared_cache_mb=0, hash_strategy="",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=1, wall_clock_s=50.0, hit_rate=0.0, miss_empty=0,
                miss_collision=0, inserted=0, similarity=1.0, diff_pct=0.0,
                max_pixel_diff=0, mean_diff=0.0
            ),
            benchmark_full.BenchResult(
                config_desc="cache64MB_blake2b_w2", shared_cache_mb=64, hash_strategy="blake2b",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=2, wall_clock_s=30.0, hit_rate=0.85, miss_empty=100,
                miss_collision=5, inserted=500, similarity=0.999, diff_pct=0.01,
                max_pixel_diff=2, mean_diff=0.001
            ),
        ]
        benchmark_full._write_csv(results, tmp_path)
        csv_path = tmp_path / "benchmark_results.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["config_desc"] == "no-cache_w1"
        assert rows[1]["shared_cache_mb"] == "64"
        assert "similarity" in rows[0]
        assert "max_pixel_diff" in rows[0]


class TestMarkdownReport:
    def test_report_generation(self, tmp_path: Path) -> None:
        results = [
            benchmark_full.BenchResult(
                config_desc="no-cache_w1", shared_cache_mb=0, hash_strategy="",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=1, wall_clock_s=100.0, hit_rate=0.0, miss_empty=0,
                miss_collision=0, inserted=0, similarity=1.0, diff_pct=0.0,
                max_pixel_diff=0, mean_diff=0.0
            ),
            benchmark_full.BenchResult(
                config_desc="cache64MB_blake2b_w2", shared_cache_mb=64, hash_strategy="blake2b",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=2, wall_clock_s=40.0, hit_rate=0.9, miss_empty=50,
                miss_collision=2, inserted=300, similarity=0.999, diff_pct=0.001,
                max_pixel_diff=1, mean_diff=0.0001
            ),
        ]
        benchmark_full._generate_markdown_report(results, tmp_path)
        md_path = tmp_path / "RELATORIO_BENCHMARK.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Relatorio Benchmark" in content
        assert "Speedup" in content
        assert "no-cache_w1" in content
        assert "cache64MB_blake2b_w2" in content


class TestBenchResult:
    def test_all_fields(self) -> None:
        r = benchmark_full.BenchResult(
            config_desc="test", shared_cache_mb=128, hash_strategy="xxhash",
            cache_key_mode="exact", cache_key_decimals=0,
            workers=4, wall_clock_s=25.0, hit_rate=0.95, miss_empty=10,
            miss_collision=1, inserted=1000, similarity=0.9999, diff_pct=0.001,
            max_pixel_diff=1, mean_diff=0.0001
        )
        assert r.config_desc == "test"
        assert r.shared_cache_mb == 128
        assert r.hash_strategy == "xxhash"
        assert r.workers == 4
        assert r.wall_clock_s == 25.0
        assert r.hit_rate == 0.95
        assert r.miss_empty == 10
        assert r.miss_collision == 1
        assert r.inserted == 1000
        assert r.similarity == 0.9999
        assert r.diff_pct == 0.001
        assert r.max_pixel_diff == 1
        assert r.mean_diff == 0.0001


class TestCompareBaseline:
    def test_identical_files(self, tmp_path: Path) -> None:
        arr = np.full((100, 100), 128, dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, arr)
        _write_tif(b, arr)
        result = benchmark_full._compare_with_baseline(a, b)
        assert result["similarity"] == pytest.approx(1.0)
        assert result["diff_pct"] == pytest.approx(0.0)
        assert result["max_pixel_diff"] == 0
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_different_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, np.full((100, 100), 50, dtype=np.uint8))
        _write_tif(b, np.full((100, 100), 200, dtype=np.uint8))
        result = benchmark_full._compare_with_baseline(a, b)
        assert result["similarity"] == pytest.approx(0.0)
        assert result["max_pixel_diff"] == 150
        assert result["mean_diff"] == pytest.approx(150.0)
        assert result["diff_pct"] == pytest.approx(100.0)


class TestEnsureSrcPath:
    def test_src_added_to_path(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        old_len = len(sys.path)
        benchmark_full._ensure_src_path(tmp_path)
        assert str(src_dir) in sys.path


class TestGeneratePlots:
    def test_plots_created(self, tmp_path: Path) -> None:
        results = [
            benchmark_full.BenchResult(
                config_desc="no-cache_w1", shared_cache_mb=0, hash_strategy="",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=1, wall_clock_s=100.0, hit_rate=0.0, miss_empty=0,
                miss_collision=0, inserted=0, similarity=1.0, diff_pct=0.0,
                max_pixel_diff=0, mean_diff=0.0
            ),
            benchmark_full.BenchResult(
                config_desc="cache64MB_blake2b_w2", shared_cache_mb=64, hash_strategy="blake2b",
                cache_key_mode="exact",
                cache_key_decimals=0,
                workers=2, wall_clock_s=40.0, hit_rate=0.9, miss_empty=50,
                miss_collision=2, inserted=300, similarity=0.999, diff_pct=0.001,
                max_pixel_diff=1, mean_diff=0.0001
            ),
        ]
        benchmark_full._generate_plots(results, tmp_path)
        plots_dir = tmp_path / "plots"
        assert plots_dir.exists()
        assert (plots_dir / "time_vs_cache.png").exists()
        assert (plots_dir / "hitrate_vs_cache.png").exists()
        assert (plots_dir / "workers_vs_time.png").exists()
        for png in plots_dir.glob("*.png"):
            assert png.stat().st_size > 1000


class TestIntegrationCLI:
    def test_help_flag(self) -> None:
        module_path = Path(benchmark_full.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(module_path), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--samples" in result.stdout
        assert "--outdir" in result.stdout

    def test_missing_input(self) -> None:
        module_path = Path(benchmark_full.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(module_path), "--input", "/nonexistent.tif"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestRoundMode:
    def test_round_decimals_in_matrix(self) -> None:
        matrix = benchmark_full._build_matrix()
        for dec in [1, 2, 3, 4, 5]:
            round_cfgs = [c for c in matrix if c["cache_key_mode"] == "round"
                          and c["cache_key_decimals"] == dec]
            assert len(round_cfgs) == 30  # 5 sizes x 2 strategies x 3 workers

    def test_round_config_desc(self) -> None:
        cfg = {"shared_cache_mb": 256, "hash_strategy": "blake2b",
               "workers": 4, "cache_key_mode": "round", "cache_key_decimals": 3}
        assert benchmark_full._config_desc(cfg) == "cache256MB_blake2b_w4_roundd3"

    def test_round_decimals_in_cmd(self) -> None:
        # Verify the matrix contains all expected decimal values
        matrix = benchmark_full._build_matrix()
        decimals = sorted(set(c["cache_key_decimals"] for c in matrix
                              if c["cache_key_decimals"] is not None))
        assert decimals == [1, 2, 3, 4, 5]
