import numpy as np
import pytest
from pipeline.signatures.key_builder import (
    prepare_for_model,
    make_cache_key_values,
    signature_bytes_for_row,
)


class TestPrepareForModel:
    def test_prepare_uint8_to_float32(self) -> None:
        arr = np.array([[1, 2, 3]], dtype=np.uint8)
        result = prepare_for_model(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr.astype(np.float32))

    def test_prepare_float32_is_no_copy_when_already_float32(self) -> None:
        arr = np.array([[1.5, 2.5]], dtype=np.float32)
        result = prepare_for_model(arr)
        assert np.shares_memory(arr, result), "already float32 should share memory"

    def test_prepare_uint16_to_float32(self) -> None:
        arr = np.array([[100, 200]], dtype=np.uint16)
        result = prepare_for_model(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr.astype(np.float32))


class TestMakeCacheKeyValues:
    def test_uint8_exact_preserves_dtype(self) -> None:
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        result = make_cache_key_values(arr, mode="exact", decimals=None)
        assert result.dtype == np.uint8
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, arr)

    def test_uint16_exact_preserves_dtype(self) -> None:
        arr = np.array([[1000, 2000]], dtype=np.uint16)
        result = make_cache_key_values(arr, mode="exact", decimals=None)
        assert result.dtype == np.uint16
        assert result.flags["C_CONTIGUOUS"]

    def test_float32_exact_preserves_small_difference(self) -> None:
        arr = np.array([[0.12345670], [0.12345671]], dtype=np.float32)
        result = make_cache_key_values(arr, mode="exact", decimals=None)
        assert result.dtype == np.float32
        row0 = signature_bytes_for_row(result, 0)
        row1 = signature_bytes_for_row(result, 1)
        assert row0 != row1, "very close float32 values must produce different keys in exact mode"

    def test_float32_round_collapses_nearby_values(self) -> None:
        arr = np.array([[0.123451], [0.123459]], dtype=np.float32)
        result = make_cache_key_values(arr, mode="round", decimals=4)
        row0 = signature_bytes_for_row(result, 0)
        row1 = signature_bytes_for_row(result, 1)
        assert row0 == row1, "nearby values rounded to 4 decimals must produce same key"

    def test_float32_round_requires_decimals_gte_1(self) -> None:
        arr = np.array([[0.12345]], dtype=np.float32)
        with pytest.raises(ValueError, match="decimals must be >= 1"):
            make_cache_key_values(arr, mode="round", decimals=0)

    def test_float64_is_converted_to_float32(self) -> None:
        arr = np.array([[1.5, 2.5]], dtype=np.float64)
        result = make_cache_key_values(arr, mode="exact", decimals=None)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr.astype(np.float32))

    def test_unsupported_dtype_raises_typeerror(self) -> None:
        arr = np.array([[1, 2]], dtype=np.int32)
        with pytest.raises(TypeError) as exc:
            make_cache_key_values(arr, mode="exact", decimals=None)
        assert "int32" in str(exc.value)

    def test_invalid_mode_raises_valueerror(self) -> None:
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError) as exc:
            make_cache_key_values(arr, mode="banana", decimals=None)
        assert "banana" in str(exc.value)

    def test_output_is_always_c_contiguous(self) -> None:
        arr_f32 = np.array([[1.5, 2.5]], dtype=np.float32)
        arr_u8 = np.array([[1, 2]], dtype=np.uint8)
        result_exact = make_cache_key_values(arr_f32, mode="exact", decimals=None)
        result_round = make_cache_key_values(arr_f32, mode="round", decimals=2)
        result_u8 = make_cache_key_values(arr_u8, mode="exact", decimals=None)
        assert result_exact.flags["C_CONTIGUOUS"]
        assert result_round.flags["C_CONTIGUOUS"]
        assert result_u8.flags["C_CONTIGUOUS"]


class TestSignatureBytes:
    def test_signature_bytes_different_pixels_different_bytes(self) -> None:
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8)
        hashes = {signature_bytes_for_row(arr, i) for i in range(4)}
        assert len(hashes) == 4, "4 different pixels should produce 4 different bytes"


class TestInitExports:
    def test_init_exports(self) -> None:
        from pipeline.signatures import make_cache_key_values as m1
        from pipeline.signatures import prepare_for_model as m2
        from pipeline.signatures import signature_bytes_for_row as m3
        assert callable(m1)
        assert callable(m2)
        assert callable(m3)
