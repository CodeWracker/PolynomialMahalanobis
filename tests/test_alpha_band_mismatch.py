"""Teste de mismatch entre bandas espectrais do TIFF e colunas do samples.txt (Goal 06.5)."""
import numpy as np
import pytest
import rasterio
from src.main import main


def test_mismatch_raises_value_error(tmp_path):
    """ValueError quando len(band_idx) != n_bands."""
    samples = tmp_path / "samples.txt"
    samples.write_text("1.0 2.0\n3.0 4.0\n")

    tif = tmp_path / "input.tif"
    arr = np.zeros((3, 10, 10), dtype=np.float32)
    with rasterio.open(str(tif), "w", driver="GTiff", count=3, dtype="float32",
                       width=10, height=10) as dst:
        dst.write(arr)
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        ]

    out = tmp_path / "out.tif"

    with pytest.raises(ValueError, match="Mismatch"):
        main(
            str(tif),
            str(samples),
            order=3,
            exp_value=-1.0,
            out_path=str(out),
            n_workers=1,
            tile_h=10,
            tile_w=10,
        )
