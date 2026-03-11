# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- None yet

### Changed
- None yet

### Fixed
- None yet

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
