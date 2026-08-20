"""Tests for src/compare/outputs.py — tile-by-tile GeoTIFF comparison."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

H, W = 300, 300


def _write_tif(path: Path, arr: np.ndarray) -> None:
    """Helper: write a single-band uint8 GeoTIFF."""
    h, w = arr.shape
    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        str(path), "w",
        driver="GTiff", dtype="uint8",
        count=1, height=h, width=w,
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(arr.astype(np.uint8), 1)


class TestIdenticalImages:
    """Two identical images must produce similarity==1.0 and is_identical==True."""

    def test_identical_images(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        arr = np.full((H, W), 128, dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, arr)
        _write_tif(b, arr)

        result = compare_outputs(str(a), str(b), tile_size=128)

        assert result.is_identical is True
        assert result.global_max_diff == 0
        assert result.global_min_diff == 0
        assert result.global_mean_diff == pytest.approx(0.0)
        assert result.global_similarity == pytest.approx(1.0)
        assert result.total_pixels == H * W
        assert result.total_diff_pixels == 0


class TestKnownDiff:
    """Verify known pixel differences are detected correctly."""

    def test_uniform_diff_all_pixels(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        a_arr = np.full((H, W), 100, dtype=np.uint8)
        b_arr = np.full((H, W), 200, dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        result = compare_outputs(str(a), str(b), tile_size=128)

        assert result.global_max_diff == 100
        assert result.global_mean_diff == pytest.approx(100.0)
        assert result.total_diff_pixels == H * W
        assert result.global_similarity == pytest.approx(0.0)


class TestUint8Underflow:
    """CRITICAL: verify int32 conversion prevents uint8 underflow."""

    def test_no_uint8_underflow(self, tmp_path: Path) -> None:
        """50 vs 200 must yield max_diff=150, NOT 62 (underflow)."""
        from src.compare.outputs import compare_outputs

        a_arr = np.full((H, W), 50, dtype=np.uint8)
        b_arr = np.full((H, W), 200, dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        result = compare_outputs(str(a), str(b), tile_size=128)

        # Underflow would give: abs(50 - 200) in uint8 = abs(106) = 106
        # or in some order: abs(200 - 50) = 150 which happens to be correct
        # But 50 - 200 in uint8 wraps: (50-200+256)%256 = 106
        # Correct: abs(50 - 200) in int32 = 150
        assert result.global_max_diff == 150, (
            f"Expected 150 but got {result.global_max_diff} — "
            "uint8 underflow likely occurred"
        )


class TestTileSizeIndependence:
    """Comparison must give same result regardless of tile_size."""

    def test_tile_size_64_vs_128(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        a_arr = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)
        b_arr = a_arr.copy()
        b_arr[100:150, 100:150] = 255
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        r64 = compare_outputs(str(a), str(b), tile_size=64)
        r128 = compare_outputs(str(a), str(b), tile_size=128)

        assert r64.global_max_diff == r128.global_max_diff
        assert r64.global_min_diff == r128.global_min_diff
        assert r64.total_pixels == r128.total_pixels
        assert r64.total_diff_pixels == r128.total_diff_pixels
        assert r64.global_similarity == pytest.approx(r128.global_similarity)
        # sum_abs_diff determines global_mean_diff
        assert r64.global_mean_diff == pytest.approx(r128.global_mean_diff)


class TestDimensionMismatch:
    """Different dimensions must raise ValueError."""

    def test_height_mismatch(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        a_arr = np.zeros((200, W), dtype=np.uint8)
        b_arr = np.zeros((300, W), dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        with pytest.raises(ValueError, match="[dD]imension"):
            compare_outputs(str(a), str(b))

    def test_width_mismatch(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        a_arr = np.zeros((H, 200), dtype=np.uint8)
        b_arr = np.zeros((H, 300), dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        with pytest.raises(ValueError, match="[dD]imension"):
            compare_outputs(str(a), str(b))


class TestTileCount:
    """Verify tile count matches ceil(h/tile_size) * ceil(w/tile_size)."""

    def test_tile_count_256(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        arr = np.zeros((256, 256), dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, arr)
        _write_tif(b, arr)

        result = compare_outputs(str(a), str(b), tile_size=128)
        expected_tiles = math.ceil(256 / 128) * math.ceil(256 / 128)
        assert len(result.tile_results) == expected_tiles

    def test_tile_count_non_divisible(self, tmp_path: Path) -> None:
        """300x300 with tile_size=128 → ceil(300/128)=3 → 9 tiles."""
        from src.compare.outputs import compare_outputs

        arr = np.zeros((H, W), dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, arr)
        _write_tif(b, arr)

        result = compare_outputs(str(a), str(b), tile_size=128)
        expected = math.ceil(H / 128) * math.ceil(W / 128)
        assert len(result.tile_results) == expected


class TestPartialDiff:
    """Correct similarity for partial pixel differences."""

    def test_partial_diff_similarity(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        a_arr = np.zeros((H, W), dtype=np.uint8)
        b_arr = np.zeros((H, W), dtype=np.uint8)
        # set exactly 1/4 of pixels different
        quarter = (H * W) // 4
        count = 0
        for i in range(H):
            for j in range(W):
                if count < quarter:
                    b_arr[i, j] = 255
                    count += 1
                else:
                    break
            if count >= quarter:
                break

        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        result = compare_outputs(str(a), str(b), tile_size=128)
        expected_sim = 1.0 - (result.total_diff_pixels / result.total_pixels)
        assert result.global_similarity == pytest.approx(expected_sim, abs=0.01)
        assert result.total_diff_pixels == quarter


class TestTileCompareResultFields:
    """Verify TileCompareResult has all required fields with correct values."""

    def test_tile_fields(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        arr = np.zeros((256, 256), dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, arr)
        _write_tif(b, arr)

        result = compare_outputs(str(a), str(b), tile_size=128)

        for tile in result.tile_results:
            assert hasattr(tile, "tile_idx")
            assert hasattr(tile, "row_off")
            assert hasattr(tile, "col_off")
            assert hasattr(tile, "height")
            assert hasattr(tile, "width")
            assert hasattr(tile, "total_pixels")
            assert hasattr(tile, "diff_pixels")
            assert hasattr(tile, "max_diff")
            assert hasattr(tile, "min_diff")
            assert hasattr(tile, "mean_diff")
            assert hasattr(tile, "sum_abs_diff")


class TestCLI:
    """Standalone CLI prints valid JSON."""

    def test_cli_output(self, tmp_path: Path) -> None:
        arr = np.full((H, W), 128, dtype=np.uint8)
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, arr)
        _write_tif(b, arr)

        mod_path = Path(__file__).parent.parent / "src" / "compare" / "outputs.py"
        result = subprocess.run(
            [sys.executable, str(mod_path), str(a), str(b)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["is_identical"] is True
        assert data["global_similarity"] == pytest.approx(1.0)


class TestGlobalMeanDiff:
    """global_mean_diff = sum_abs_diff / total_pixels."""

    def test_global_mean_sum(self, tmp_path: Path) -> None:
        from src.compare.outputs import compare_outputs

        a_arr = np.ones((H, W), dtype=np.uint8) * 10
        b_arr = np.ones((H, W), dtype=np.uint8) * 20
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tif(a, a_arr)
        _write_tif(b, b_arr)

        result = compare_outputs(str(a), str(b), tile_size=64)

        expected_mean = 10.0
        assert result.global_mean_diff == pytest.approx(expected_mean)
        expected_sum = H * W * 10
        assert result.global_mean_diff * result.total_pixels == pytest.approx(expected_sum)
