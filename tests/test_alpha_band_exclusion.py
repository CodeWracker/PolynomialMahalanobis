"""Testes para detecção e exclusão automática de banda alpha (Goal 06.5)."""
import pytest
import rasterio
from unittest.mock import MagicMock
from pipeline.main import resolve_band_indices


def test_resolve_excludes_alpha_band():
    src = MagicMock()
    src.count = 5
    src.colorinterp = [
        rasterio.enums.ColorInterp.red,
        rasterio.enums.ColorInterp.green,
        rasterio.enums.ColorInterp.gray,
        rasterio.enums.ColorInterp.gray,
        rasterio.enums.ColorInterp.alpha,
    ]
    result = resolve_band_indices(src)
    assert result == [1, 2, 3, 4]


def test_resolve_returns_all_when_no_alpha():
    src = MagicMock()
    src.count = 3
    src.colorinterp = [
        rasterio.enums.ColorInterp.red,
        rasterio.enums.ColorInterp.green,
        rasterio.enums.ColorInterp.blue,
    ]
    result = resolve_band_indices(src)
    assert result == [1, 2, 3]


def test_resolve_with_alpha_first_band():
    src = MagicMock()
    src.count = 2
    src.colorinterp = [
        rasterio.enums.ColorInterp.alpha,
        rasterio.enums.ColorInterp.gray,
    ]
    result = resolve_band_indices(src)
    assert result == [2]


def test_resolve_empty_when_all_alpha():
    src = MagicMock()
    src.count = 1
    src.colorinterp = [rasterio.enums.ColorInterp.alpha]
    result = resolve_band_indices(src)
    assert result == []
