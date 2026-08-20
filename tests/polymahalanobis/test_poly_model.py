"""Smoke tests for the published polymahalanobis library.

These import the package the same way an external consumer of the PyPI
package would (`from polymahalanobis import ...`), with no sys.path hacks
and no dependency on the rest of the workspace (pipeline, cache, rasterio).
"""
import numpy as np
import pytest

from polymahalanobis import LevelBasis, PolyMahalanobis


@pytest.fixture
def samples_file(tmp_path):
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=100.0, scale=10.0, size=(50, 3))
    path = tmp_path / "samples.txt"
    np.savetxt(path, samples)
    return path


def test_makespace_builds_at_least_one_level(samples_file):
    model = PolyMahalanobis(str(samples_file), num_levels=3)
    model.makeSpace()

    assert model.n_bands == 3
    assert len(model.levels) >= 1
    assert all(isinstance(level, LevelBasis) for level in model.levels)


def test_evaluate_returns_expected_shape(samples_file):
    model = PolyMahalanobis(str(samples_file), num_levels=3)
    model.makeSpace()

    pixels = np.array([[100.0, 100.0, 100.0], [500.0, 500.0, 500.0]], dtype=np.float32)
    distances = model.evaluate(pixels)

    assert distances.shape == (2, len(model.levels))
    # An out-of-distribution sample must be at least as far as an in-distribution one.
    assert distances[1, -1] >= distances[0, -1]


def test_evaluate_single_matches_batched(samples_file):
    model = PolyMahalanobis(str(samples_file), num_levels=2)
    model.makeSpace()

    pixel = np.array([110.0, 95.0, 105.0], dtype=np.float32)
    single = model.evaluate_single(pixel)
    batched = model.evaluate(pixel.reshape(1, -1))[0]

    np.testing.assert_allclose(single, batched)


def test_evaluate_rejects_wrong_band_count(samples_file):
    model = PolyMahalanobis(str(samples_file), num_levels=2)
    model.makeSpace()

    with pytest.raises(ValueError):
        model.evaluate(np.array([[1.0, 2.0]], dtype=np.float32))
