# LX Anonymizer

LX Anonymizer is a toolkit for de-identifying endoscopy frames and medical reports. It combines OCR pipelines, spaCy-based NER, heuristic sanitizers, and report-specific rules to redact or pseudonymize sensitive information while preserving clinical context.

## Highlights
- **End-to-end anonymization** of PDFs and frame sequences using OCR, NER, and pseudonymization helpers.
- **Modular pipeline** that lets you choose between Tesseract, TrOCR, ensemble OCR, and multiple metadata extractors.
- **Human-in-the-loop ready** outputs: original/anonymized text side by side, metadata JSON, and validation artefacts.
- **Extensible ruleset** covering device-specific renderers, fuzzy name matching, and language-specific replacements.

## Requirements
- Python 3.12+
- Linux or macOS (Windows support is experimental)
- NVIDIA GPU recommended for real-time video anonymization (CUDA 12.x). CPU-only processing works but is slower.
- Optional extras:
  - spaCy `de_core_news_lg` model (download after installation)
  - Torch vision/audio for video OCR workloads
  - Ollama-compatible LLMs for advanced metadata extraction

## Installation

### From PyPI *(upcoming release)*
```bash
pip install lx-anonymizer
```

Install extras to tailor the footprint:
```bash
pip install "lx-anonymizer[gpu,ocr,llm,dev]"
```

### From source
```bash
git clone https://github.com/wg-lux/lx-anonymizer.git
cd lx-anonymizer
uv sync
```

### Nix development shell
```bash
direnv allow
nix develop
```
This loads GPU, OCR, and tooling dependencies declared in `devenv.nix`.

## Model downloads
After installation, fetch the German spaCy model:
```bash
python -m spacy download de_core_news_lg
```
First CLI runs also download OCR checkpoints (EAST, TrOCR, etc.). For air-gapped deployments, grab the archives listed in [`lx_anonymizer/settings.py`](lx_anonymizer/settings.py) and place them in `~/.cache/lx-anonymizer`.

## Quickstart

### CLI
```bash
python -m cli.report_reader process report.pdf --ensemble --output-dir ./anonymized
```
Useful options:
- `--llm-extractor {deepseek,medllama,llama3}` for LLM-powered metadata extraction.
- `--use-ocr` and `--ensemble` to switch OCR strategies.
- `batch` and `extract` sub-commands for folder processing or metadata-only runs.

### Python API
```python
from lx_anonymizer import ReportReader

reader = ReportReader(locale="de_DE")
original, anonymized, meta = reader.process_report(
    pdf_path="/path/to/report.pdf",
    use_ensemble=True,
    use_llm_extractor="deepseek",
)
```
See [`tests/test_cli_integration.py`](tests/test_cli_integration.py) for more examples.

## Data directories
By default, outputs live in `~/etc/lx-anonymizer/{data,temp}`. Adjust them in [`lx_anonymizer/directory_setup.py`](lx_anonymizer/directory_setup.py). Clean `temp` regularly to avoid large intermediate artefacts.

## Development workflow
- Format & lint: `uv run flake8`
- Tests (CPU friendly): `uv run pytest -m "not gpu"`
  - GPU tests are marked and can be run with `-m gpu`
- Build wheel for release: `uv run python -m build`
- Full local check helper: `scripts/run_checks.sh`

## Project roadmap
1. Publish CPU-only wheel to TestPyPI.
2. Add optional extras for GPU/LLM workloads and slim default install.
3. Automate release workflow (wheel + sdist upload, GitHub release notes).
4. Expose REST/gRPC service with validation UI.

## Contributing
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines, testing instructions, and communication channels.

## License
Released under the [MIT License](LICENSE).

## Contact
Questions? Email lux@coloreg.de .



