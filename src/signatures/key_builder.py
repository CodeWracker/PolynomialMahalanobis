"""Cache key construction and signature utilities for PolyModel evaluation."""

import numpy as np


def prepare_for_model(signatures: np.ndarray) -> np.ndarray:
    """Convert signatures to float32 for PolyModel evaluation.

    Args:
        signatures: Input array of any numeric dtype.

    Returns:
        Array converted to float32 (no copy if already float32).

    Raises:
        N/A: This function does not raise any exceptions.
    """
    return signatures.astype(np.float32, copy=False)


def make_cache_key_values(values: np.ndarray, mode: str, decimals: int | None) -> np.ndarray:
    """Build cache key array from pixel values with dtype-aware handling.

    Args:
        values: Input array of pixel signatures.
        mode: Cache key mode.
            'exact' preserves minimal differences between float32 values
            (e.g., 0.12345670 != 0.12345671).
            'round' rounds to N decimal places, collapsing nearby values
            into the same key to increase cache hit rate.
        decimals: Number of decimal places for rounding (required when mode='round').

    Returns:
        C-contiguous array suitable for key construction.

    Raises:
        ValueError: If mode is not supported.
        TypeError: If dtype is not supported.
    """
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)

    if arr.dtype == np.float32:
        if mode == "exact":
            return np.ascontiguousarray(arr.astype("<f4", copy=False))
        if mode == "round":
            if decimals is None or decimals < 1:
                raise ValueError("decimals must be >= 1 for round mode")
            rounded = np.round(arr, decimals=decimals).astype(np.float32)
            return np.ascontiguousarray(rounded.astype("<f4", copy=False))
        raise ValueError(f"unsupported cache key mode: {mode}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr.astype("u1", copy=False))

    if arr.dtype == np.uint16:
        return np.ascontiguousarray(arr.astype("<u2", copy=False))

    raise TypeError(f"unsupported dtype for cache key: {arr.dtype}")


def signature_bytes_for_row(key_array: np.ndarray, row_idx: int) -> bytes:
    """Extract raw bytes of a signature row for hashing.

    Args:
        key_array: 2D array of cache key values.
        row_idx: Row index to extract.

    Returns:
        Bytes of the row.

    Raises:
        IndexError: If row_idx is out of bounds for key_array.
    """
    result = key_array[row_idx].tobytes()
    return result
