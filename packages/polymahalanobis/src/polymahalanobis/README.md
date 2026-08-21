# polymahalanobis

A polynomial Mahalanobis distance classifier for multivariate / multispectral data.

Given a set of reference samples, `PolyMahalanobis` builds a nested sequence of
polynomial-expanded subspaces (via SVD) and evaluates how far a new sample is
from that reference distribution. Useful for anomaly detection, similarity
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

## Reference samples format

`PolyMahalanobis(sample_file, num_levels)` loads its reference samples from a
plain text file using `numpy.loadtxt`. The rules:

- One sample per line, band values separated by whitespace (spaces or tabs).
- Every line must have the same number of columns. That column count becomes
  `n_bands`, and it must match the last dimension of every array you later
  pass to `evaluate`, `evaluate_single`, or `evaluate_image`.
- Values are parsed as floating point numbers.
- Lines starting with `#` are treated as comments and skipped.
- No header row.

Example, for 3-band samples (say, R, G, B pixel values):

```
93.0 141.0 108.0
95.5 138.2 110.1
90.0 145.0 105.5
```

These lines should be a curated set of reference observations that define
what "normal" looks like for your use case (for example, healthy crop
pixels, or measurements from a known material). `makeSpace()` fits the
polynomial subspaces to them, and `evaluate()` then reports how far new
observations are from that reference distribution. A few dozen samples is
usually a reasonable minimum; too few samples relative to `num_levels` can
leave the later polynomial levels without enough variance to be meaningful.

## References

This implementation is based on the Polynomial Mahalanobis Distance metric
introduced in:

- G. Grudic and J. Mulligan, "Outdoor Path Labeling Using Polynomial
  Mahalanobis Distance," *Robotics: Science and Systems II*, 2006.
  https://doi.org/10.15607/RSS.2006.II.020
- A. C. Sobieranski, D. D. Abdala, E. Comunello, and A. von Wangenheim,
  "Learning a color distance metric for region-based image segmentation,"
  *Pattern Recognition Letters*, vol. 30, no. 16, pp. 1496-1506, 2009.
  https://doi.org/10.1016/j.patrec.2009.08.002

## Full project

This library is the core algorithm extracted from the
[PolynomialMahalanobis](https://github.com/CodeWracker/PolynomialMahalanobis)
project, which also includes a parallelized command-line pipeline for
classifying multispectral GeoTIFF orthophotos. That pipeline is not part of
this PyPI package; see the repository for it.
