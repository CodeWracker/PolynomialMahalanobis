from pathlib import Path
import numpy as np

from tests.helpers import run_pipeline, write_synthetic_tif
from src.compare.outputs import compare_outputs


def test_tiny_cache_end_to_end_matches_no_cache(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
    input_tif = tmp_path / "in.tif"
    write_synthetic_tif(input_tif, arr)
    samples = tmp_path / "samples.txt"
    samples.write_text("10 20 30\n200 210 220\n")
    log = tmp_path / "run.log"

    out_nocache = tmp_path / "out_nocache.tif"
    run_pipeline(input_tif, samples, out_nocache, log,
                order=1, exp=-1.0, tile_size=16, workers=1, shared_cache_mb=0)

    out_tinycache = tmp_path / "out_tinycache.tif"
    # ~3 bandas uint8 -> slot_size=5 bytes. 0.0001 MB ~ 104 bytes ~ 20 slots,
    # bem menos que as assinaturas \u00fanicas esperadas em uma imagem 16x16 aleat\u00f3ria.
    run_pipeline(input_tif, samples, out_tinycache, log,
                order=1, exp=-1.0, tile_size=16, workers=1,
                shared_cache_mb=0.0001, cache_key_mode="exact")

    result = compare_outputs(out_nocache, out_tinycache)
    assert result.is_identical, (
        f"Cache pequeno/cheio n\u00e3o deve alterar resultado. max_diff={result.global_max_diff}"
    )
