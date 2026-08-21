# Changelog

All notable changes to the `polymahalanobis` package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.1] - 2026-08-20

### Changed

- README: documented the reference samples file format (`samples.txt`),
  including the `numpy.loadtxt` parsing rules and an example.
- README: added a References section citing the two papers this
  implementation is based on (Grudic & Mulligan, 2006; Sobieranski et al.,
  2009).

## [0.1.0] - 2026-08-20

### Added

- Initial extraction of `PolyMahalanobis` into a standalone, PyPI-publishable
  package (`packages/polymahalanobis`), depending only on `numpy`.
- Repository reorganized as a uv workspace: `packages/polymahalanobis` (the
  published library) and `apps/pipeline` (the internal orthophoto CLI
  pipeline, unchanged in behavior, not published).
