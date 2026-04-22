# Changelog

All notable changes to cornetto are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-04-22

### Added
- `delta_mode` plotting: subtracts truths from samples.
- `tick_rotation` kwarg on `corner()` / `quick_corner()`.

### Changed
- `MaxNLocator` now uses `prune="lower"` instead of `prune="both"` on corner
  tick axes, keeping the upper (more informative) endpoint.

### Fixed
- Bottom-row xtick labels going missing on large corners: using
  `set_xticklabels([])` on the diagonal was clobbering the shared-axis
  formatter.

## [0.2.1] — 2026-04-22

### Added
- GitHub Actions workflow running the test suite with coverage reporting.
- `jax` (optional) dependency for accelerated KDE computations.
- `quick_corner` now accepts `kwargs_truths` and styles truth overlays the
  same way as `corner()`.

### Changed
- Expanded test coverage and README (badges, usage notes).
- Reorganized assets: README badge order and corner-plot image source updated accordingly.

## [0.2.0] — 2026-04-21

First public release.

[0.2.2]: https://github.com/nicoborghi/cornetto/releases/tag/v0.2.2
[0.2.1]: https://github.com/nicoborghi/cornetto/releases/tag/v0.2.1
[0.2.0]: https://github.com/nicoborghi/cornetto/releases/tag/v0.2.0
