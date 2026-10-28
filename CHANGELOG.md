# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- GitHub Actions native wheel build and release pipeline for Linux and macOS
- PyPI trusted-publishing based release workflow
- Import smoke test coverage for core package modules

### Changed
- Prepare candidate `1.0.9.dev1` with the report anonymization v2 contract; require
  `lx-dtypes>=0.5.4.dev1` and coordinate the endoreg-db consumer release.
- Preserve valid-empty provider fallback opportunities and distinguish extraction
  evidence from Rust-owned durable lifecycle state.
- README and contributing docs to match native-wheel CI and flake/devenv packaging
- Local release guidance to treat `python -m build --sdist` as sdist validation while CI produces wheels

### Fixed
- Remove document content and raw dependency exception payloads from report
  diagnostics, including worker-captured streams and public failure tracebacks.
- Frame removal no longer re-times retained frames (`setpts=N/FRAME_RATE/TB`,
  `asetpts=N/SR/TB`). Retained video frames and audio samples keep their decoded
  timestamps and removed spans remain as gaps, so recorded presentation
  timestamps can be matched against external clocks. endoreg-db can drop its
  host-side retiming shim once it pins a release containing this change.
- Standalone frame-removal commands (`FrameCleaner` direct and CPU fallback,
  `VideoProcessor.remove_frames_streaming`) pass `-copyts`, `-fps_mode passthrough`,
  `-enc_time_base demux` and an MP4 track timescale equal to the source time base,
  so retained frames keep their exact recorded ticks without host hooks. A missing
  or unrepresentable source time base is logged and leaves the timescale to the
  muxer, so frame removal never fails on it.

## [0.9.0.4] - 2026-03-11
### Added
- Packaged console entry point `lx-anonymizer`
- Optional extras split for OCR, LLM, Flair NER, Django, and development tooling

### Changed
- Slimmed default dependency set for PyPI installs
- Switched base OpenCV dependency to `opencv-python-headless`
- Updated README and release instructions to match shipped extras and CLI

### Fixed
- Removed EAST model download on import
- Avoided heavyweight OCR/NER initialization during package import
- Excluded generated reports, backup files, and unused scripts from release artifacts

## [0.8.0] - 2024-12-30
### Added
- CLI (`cli.report_reader`) supporting single, batch, and metadata extraction commands
- Ensemble OCR pipeline combining Tesseract, TrOCR, and heuristic cleanup
- LLM-assisted metadata extraction via Ollama backend
- Extensive test suite covering CLI, OCR diagnostics, and anonymization helpers

### Changed
- Improved directory setup for data/temp storage outside the Nix store

### Fixed
- Resolved various frame anonymization and metadata parsing bugs (see logs prior to 2025-01)

## [0.7.x] - 2024-10-15
### Added
- Initial publication of lx_anonymizer package with OCR pipeline and pseudonymization utilities

### Changed
- Iterative improvements to spaCy extractors and device-specific renderers

### Fixed
- Stabilized TesserOCR fallback and directory creation errors

[Unreleased]: https://github.com/wg-lux/lx-anonymizer/compare/v0.9.0.4...HEAD
[0.9.0.4]: https://github.com/wg-lux/lx-anonymizer/releases/tag/v0.9.0.4
[0.8.0]: https://github.com/wg-lux/lx-anonymizer/releases/tag/v0.8.0
