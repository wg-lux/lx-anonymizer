# Contributing to LX Anonymizer

Thanks for taking the time to contribute! This guide explains how to propose improvements, report bugs, and ship new features safely.

## Table of Contents
- [Project scope](#project-scope)
- [Ways to help](#ways-to-help)
- [Development workflow](#development-workflow)
- [Pull request checklist](#pull-request-checklist)
- [Communication](#communication)
- [Community standards](#community-standards)

## Project scope
LX Anonymizer focuses on anonymizing medical reports and endoscopy frames via OCR, NLP, and heuristic rules. Core areas include:
- OCR pipelines (Tesseract, TrOCR, ensemble, GPU acceleration)
- Named-entity recognition and pseudonymization
- CLI tooling and batch processing
- Integration with validation and metadata export flows

If a proposal falls outside these boundaries, consider opening a discussion first to align direction.

## Ways to help
- **Report issues:** Include environment details, reproducible steps, and logs when possible.
- **Improve docs:** Clarify setup instructions, add examples, enhance troubleshooting.
- **Add tests:** Cover edge cases (empty PDFs, low-quality scans, GPU-disabled runs, etc.).
- **Optimize pipelines:** Reduce inference time, memory footprint, or model download overheads.

## Development workflow
1. **Set up the environment**
   ```bash
   git clone https://github.com/wg-lux/lx-anonymizer.git
   cd lx-anonymizer
   uv sync
   # optional: direnv allow && nix develop for GPU tooling
   ```
2. **Install extras**
   - CPU-only work: `uv pip install -e .[dev,nlp,ocr]`
   - GPU pipelines: add `[gpu]`
   - LLM features: add `[llm]`
3. **Run checks**
   ```bash
   ./scripts/run_checks.sh
   ```
   The script installs `uv` if needed, syncs dependencies, lints, and runs the CPU-friendly test suite. Pass extra pytest arguments as desired, e.g. `./scripts/run_checks.sh -k anonymizer`.
   GPU/LLM tests are behind markers (`-m gpu`, `-m llm`) to keep defaults lightweight.
4. **Keep commits focused**
   - Separate concerns (docs vs code changes) to simplify review.
   - Reference issue numbers in commit messages when relevant.

## Pull request checklist
- [ ] Tests pass locally (`pytest -m "not gpu"` at minimum)
- [ ] Lint passes (`flake8`)
- [ ] New/changed functionality is documented (`README.md`, docstrings, or examples)
- [ ] Added tests for new behavior or regression fixes
- [ ] Updated changelog entry under the “Unreleased” section
- [ ] Confirmed large assets aren’t committed (logs, model weights, datasets)

CI will run linting, tests, and build the wheel/tarball automatically. Fix any reported regressions before requesting review.

## Communication
- **Issues:** https://github.com/wg-lux/lx-anonymizer/issues
- **Security disclosures:** Please email [anonymizer@wg-lux.org](mailto:anonymizer@wg-lux.org) instead of filing a public issue.
- **Discussions:** Use GitHub Discussions (to be enabled) for roadmap ideas and architecture deep dives.

## Community standards
All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md). Respectful collaboration is non-negotiable.

Appreciate your help in keeping LX Anonymizer robust and reliable! 🙌
