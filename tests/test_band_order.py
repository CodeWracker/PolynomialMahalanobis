import numpy as np
from pathlib import Path

from tests.helpers import run_pipeline, write_synthetic_tif
from pipeline.compare.outputs import compare_outputs


def test_band_order_is_respected_not_corrected(tmp_path: Path) -> None:
    """
    Mesmo arquivo de amostras, duas entradas com bandas em ordem f\u00edsica diferente
    (uma \u00e9 o espelho da outra). Se a pipeline reordenar automaticamente, os outputs
    seriam id\u00eanticos. Eles N\u00c3O devem ser id\u00eanticos — prova que n\u00e3o há reordena\u00e7\u00e3o.
    """
    arr_normal = np.array([
        [[50, 50], [200, 200]],
        [[10, 10], [5, 5]],
    ], dtype=np.uint8)
    arr_swapped = arr_normal[::-1].copy()  # inverte a ordem f\u00edsica das bandas

    samples = tmp_path / "samples.txt"
    samples.write_text("50 10\n200 5\n")  # ordem [banda_A, banda_B], casa com arr_normal

    in_normal = tmp_path / "normal.tif"
    in_swapped = tmp_path / "swapped.tif"
    write_synthetic_tif(in_normal, arr_normal)
    write_synthetic_tif(in_swapped, arr_swapped)

    out_normal = tmp_path / "out_normal.tif"
    out_swapped = tmp_path / "out_swapped.tif"
    log = tmp_path / "run.log"

    run_pipeline(in_normal, samples, out_normal, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0)
    run_pipeline(in_swapped, samples, out_swapped, log,
                order=1, exp=-1.0, tile_size=4, workers=1, shared_cache_mb=0)

    result = compare_outputs(out_normal, out_swapped)
    assert not result.is_identical, (
        "Pipeline produziu o MESMO resultado para bandas em ordem f\u00edsica diferente "
        "com o mesmo arquivo de amostras — isso indicaria reordena\u00e7\u00e3o autom\u00e1tica indevida."
    )
