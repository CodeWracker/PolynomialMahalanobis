import numpy as np
import pytest
from unittest.mock import MagicMock

from src.main import validate_and_resolve_dtype


def make_fake_src(dtypes_per_band: list[str]) -> MagicMock:
    src = MagicMock()
    src.dtypes = dtypes_per_band
    return src


def test_uint8_is_supported() -> None:
    src = make_fake_src(["uint8", "uint8", "uint8"])
    dtype, needs_cast = validate_and_resolve_dtype(src, [1, 2, 3])
    assert dtype == np.uint8
    assert not needs_cast


def test_uint16_is_supported() -> None:
    src = make_fake_src(["uint16", "uint16"])
    dtype, needs_cast = validate_and_resolve_dtype(src, [1, 2])
    assert dtype == np.uint16
    assert not needs_cast


def test_float32_is_supported() -> None:
    src = make_fake_src(["float32"])
    dtype, needs_cast = validate_and_resolve_dtype(src, [1])
    assert dtype == np.float32
    assert not needs_cast


def test_float64_is_converted_to_float32_with_cast_flag() -> None:
    src = make_fake_src(["float64"])
    dtype, needs_cast = validate_and_resolve_dtype(src, [1])
    assert dtype == np.float32
    assert needs_cast is True


def test_int32_raises_explicit_error() -> None:
    src = make_fake_src(["int32"])
    with pytest.raises(ValueError, match="não suportado"):
        validate_and_resolve_dtype(src, [1])


def test_mixed_dtypes_across_bands_raises() -> None:
    src = make_fake_src(["uint8", "uint16", "uint8"])
    with pytest.raises(ValueError, match="dtypes diferentes"):
        validate_and_resolve_dtype(src, [1, 2, 3])
