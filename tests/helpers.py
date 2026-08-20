"""
helpers.py — Utilitários compartilhados pelos testes de integração.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None  # type: ignore[assignment]

ROOT: Path = Path(__file__).parent.parent


def write_synthetic_tif(
    path: str | Path,
    array: np.ndarray,
    mask: np.ndarray | None = None,
    nodata: int | float | None = None,
) -> None:
    """
    Escreve um GeoTIFF sintético.

    Args:
        path: caminho do arquivo de saída.
        array: (n_bands, h, w) ou (h, w) para 1 banda.
        mask: (h, w) uint8, 255=válido 0=inválido. Opcional.
        nodata: valor de nodata a registrar no profile. Opcional.
    """
    if rasterio is None:
        raise ImportError("rasterio is required for write_synthetic_tif")

    if array.ndim == 2:
        array = array[None, ...]
    n_bands: int = array.shape[0]
    h: int = array.shape[1]
    w: int = array.shape[2]
    profile: dict[str, Any] = dict(
        driver="GTiff", dtype=str(array.dtype), width=w, height=h, count=n_bands,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        for b in range(n_bands):
            dst.write(array[b], b + 1)
        if mask is not None:
            dst.write_mask(mask)


def run_pipeline(
    input_tif: str | Path,
    samples_txt: str | Path,
    output_tif: str | Path,
    log: str | Path,
    timeout: int = 600,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """
    Roda o pipeline (pipeline.main) via subprocess com PYTHONHASHSEED=42 fixo.

    kwargs são convertidos em flags --kwarg-name valor (underscores -> hífen).
    Flags booleanas (True) são passadas sem valor; False é omitida.
    """
    cmd: list[str] = [
        sys.executable, "-m", "pipeline.main",
        str(input_tif), str(output_tif), str(samples_txt), str(log),
    ]
    for pair in kwargs.items():
        key: str = pair[0]
        value = pair[1]
        flag: str = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]

    env: dict[str, str] = {**os.environ, "PYTHONHASHSEED": "42"}
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )
    assert result.returncode == 0, (
        f"main.py falhou (returncode={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result
