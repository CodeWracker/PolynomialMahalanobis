# polymahalanobis

A polynomial Mahalanobis distance classifier for multivariate / multispectral data.

Given a set of reference samples, `PolyMahalanobis` builds a nested sequence of
polynomial-expanded subspaces (via SVD) and evaluates how far a new sample is
from that reference distribution — useful for anomaly detection, similarity
scoring, and multispectral classification (e.g. vegetation health from
satellite/drone imagery bands).

Only dependency: `numpy`.

## Install

```bash
pip install polymahalanobis
```

## Usage

```python
import numpy as np
from polymahalanobis import PolyMahalanobis

# samples.txt: one reference sample per line, band values space-separated
model = PolyMahalanobis("samples.txt", num_levels=3)
model.makeSpace()

new_pixels = np.array([[120.0, 45.0, 200.0]], dtype=np.float32)
distances = model.evaluate(new_pixels)  # shape (N, num_levels)

final_distance = distances[:, -1]  # accumulated distance, last level
```

- `evaluate(spectral_values)`: batched evaluation, `(N, n_bands) -> (N, num_levels)`.
- `evaluate_single(spectral_value)`: convenience for a single `(n_bands,)` sample.
- `evaluate_image(image_multispectral)`: convenience for a full `(H, W, n_bands)` array,
  with de-duplication of repeated pixel values.

## Full project

This library is the core algorithm extracted from the
[PolynomialMahalanobis](https://github.com/CodeWracker/PolynomialMahalanobis)
project, which also includes a parallelized command-line pipeline for
classifying multispectral GeoTIFF orthophotos. That pipeline is not part of
this PyPI package — see the repository for it.
