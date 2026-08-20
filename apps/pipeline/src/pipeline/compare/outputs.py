"""Tile-by-tile comparison of GeoTIFF outputs."""

import argparse
import warnings
import json
import math
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


import numpy as np


@dataclass
class TileCompareResult:
    """Per-tile diff metrics."""
    tile_idx: int
    row_off: int
    col_off: int
    height: int
    width: int
    total_pixels: int
    diff_pixels: int
    max_diff: int
    min_diff: int
    mean_diff: float
    sum_abs_diff: int


@dataclass
class GlobalCompareResult:
    """Aggregated comparison across all tiles."""
    global_max_diff: int
    global_min_diff: int
    global_mean_diff: float
    global_similarity: float
    total_pixels: int
    total_diff_pixels: int
    tile_results: list[TileCompareResult] = field(default_factory=list)

    @property
    def is_identical(self) -> bool:
        return self.global_max_diff == 0


def compare_outputs(path_a: str | Path, path_b: str | Path, tile_size: int = 1024, band: int = 1) -> GlobalCompareResult:
    """Compare two GeoTIFFs tile-by-tile.

    Raises:
        ImportError: if rasterio is not available.
        ValueError: if dimensions of the two images diverge.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError(f"compare_outputs requires rasterio: {exc}") from exc

    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        if src_a.height != src_b.height or src_a.width != src_b.width:
            raise ValueError(
                f"Dimension mismatch: A=({src_a.width}x{src_a.height}) "
                f"B=({src_b.width}x{src_b.height})"
            )

        h, w = src_a.height, src_a.width
        tile_results: list[TileCompareResult] = []
        total_pixels = 0
        total_diff_pixels = 0
        total_sum_abs_diff = 0
        global_max = 0
        global_min = 0
        first_tile = True

        rows = math.ceil(h / tile_size)
        cols = math.ceil(w / tile_size)

        tile_idx = 0
        for r in range(rows):
            for c in range(cols):
                row_off = r * tile_size
                col_off = c * tile_size
                win_h = min(tile_size, h - row_off)
                win_w = min(tile_size, w - col_off)
                window = Window(col_off, row_off, win_w, win_h)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*setting the shape.*",
                        category=DeprecationWarning,
                    )
                    tile_a = src_a.read(band, window=window)
                    tile_b = src_b.read(band, window=window)

                # CRITICAL: convert to int32 before subtraction to avoid uint8 underflow
                diff = np.abs(tile_a.astype(np.int32) - tile_b.astype(np.int32))

                t_pixels = int(win_h * win_w)
                d_pixels = int(np.count_nonzero(diff))
                s_abs = int(np.sum(diff))
                t_max = int(np.max(diff))
                t_min = int(np.min(diff))
                t_mean = float(np.mean(diff))

                tile_results.append(TileCompareResult(
                    tile_idx=tile_idx,
                    row_off=row_off,
                    col_off=col_off,
                    height=int(win_h),
                    width=int(win_w),
                    total_pixels=t_pixels,
                    diff_pixels=d_pixels,
                    max_diff=t_max,
                    min_diff=t_min,
                    mean_diff=t_mean,
                    sum_abs_diff=s_abs,
                ))

                total_pixels += t_pixels
                total_diff_pixels += d_pixels
                total_sum_abs_diff += s_abs

                if first_tile:
                    global_max = t_max
                    global_min = t_min
                    first_tile = False
                else:
                    global_max = max(global_max, t_max)
                    global_min = min(global_min, t_min)

                tile_idx += 1

    global_mean = float(total_sum_abs_diff / total_pixels) if total_pixels else 0.0
    similarity = 1.0 - (total_diff_pixels / total_pixels) if total_pixels else 1.0

    return GlobalCompareResult(
        global_max_diff=global_max,
        global_min_diff=global_min,
        global_mean_diff=global_mean,
        global_similarity=similarity,
        total_pixels=total_pixels,
        total_diff_pixels=total_diff_pixels,
        tile_results=tile_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two GeoTIFFs tile-by-tile")
    parser.add_argument("path_a", help="First GeoTIFF")
    parser.add_argument("path_b", help="Second GeoTIFF")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--band", type=int, default=1)
    args = parser.parse_args()

    result = compare_outputs(args.path_a, args.path_b, args.tile_size, args.band)
    out = asdict(result)
    out["is_identical"] = result.is_identical
    # Replace list of dataclass dicts with a count to avoid huge JSON
    out["tile_results"] = [asdict(t) for t in result.tile_results]
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.info(json.dumps({k: v for k, v in out.items()}, default=str))
