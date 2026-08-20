# Changelog

All notable changes to the `polymahalanobis` package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - Unreleased

### Added

- Initial extraction of `PolyMahalanobis` into a standalone, PyPI-publishable
  package (`packages/polymahalanobis`), depending only on `numpy`.
- Repository reorganized as a uv workspace: `packages/polymahalanobis` (the
  published library) and `apps/pipeline` (the internal orthophoto CLI
  pipeline, unchanged in behavior, not published).
