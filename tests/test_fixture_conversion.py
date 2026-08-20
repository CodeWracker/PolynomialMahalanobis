import numpy as np
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
TEST_CASES = Path(__file__).parent.parent / "test_cases"


def test_rgb_conversion_is_inverse_of_bgr() -> None:
    original = np.loadtxt(TEST_CASES / "12003.txt")
    converted = np.loadtxt(FIXTURES / "12003.txt")
    np.testing.assert_array_equal(converted[:, 0], original[:, 2])
    np.testing.assert_array_equal(converted[:, 1], original[:, 1])
    np.testing.assert_array_equal(converted[:, 2], original[:, 0])
