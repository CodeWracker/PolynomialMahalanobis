import numpy as np
import pytest
from src.main import distances_to_uint8


def test_distance_zero_gives_max_similarity_min_value() -> None:
    """distância 0 -> exp(0)=1 -> sim=1 -> value=0 (pixel mais parecido com a amostra)."""
    out = distances_to_uint8(np.array([0.0], dtype=np.float32), exp_value=-1.0)
    assert out[0] == 0


def test_large_distance_gives_value_close_to_255() -> None:
    """distância grande -> exp(-1*grande)~0 -> sim~0 -> value~255."""
    out = distances_to_uint8(np.array([50.0], dtype=np.float32), exp_value=-1.0)
    assert out[0] == 255


def test_output_dtype_is_uint8() -> None:
    out = distances_to_uint8(np.array([1.0, 2.0], dtype=np.float32), exp_value=-1.0)
    assert out.dtype == np.uint8


def test_known_value_matches_manual_calc() -> None:
    d = np.array([0.6931471805599453], dtype=np.float32)  # ln(2)
    out = distances_to_uint8(d, exp_value=-1.0)
    # sim = exp(-1 * ln2) = 0.5 -> value = round(255 - 0.5*255) = round(127.5) = 128
    expected = int(np.rint(255.0 - 0.5 * 255.0))
    assert out[0] == expected


def test_output_always_in_valid_range() -> None:
    rng = np.random.default_rng(0)
    distances = rng.uniform(0, 1000, size=1000).astype(np.float32)
    out = distances_to_uint8(distances, exp_value=-1.0)
    assert out.min() >= 0
    assert out.max() <= 255


def test_positive_exp_value_does_not_crash() -> None:
    """exp_value positivo é permitido pela CLI (não validado), só não é o uso recomendado."""
    out = distances_to_uint8(np.array([1.0], dtype=np.float32), exp_value=1.0)
    assert out.dtype == np.uint8
