# LX Anonymizer


[![Built with devenv](https://devenv.sh/assets/devenv-badge.svg)](https://devenv.sh)

LX Anonymizer is a comprehensive toolkit for de-identifying endoscopy frames and medical reports. It combines advanced OCR pipelines, spaCy-based NER, heuristic sanitizers, and report-specific rules to redact or pseudonymize sensitive information while preserving clinical context.

## Core Components

### ReportReader
Specialized for medical report anonymization with support for:
- **Multi-format processing**: PDFs and images with automatic OCR fallback
- **Advanced metadata extraction**: LLM-powered extraction using DeepSeek, MedLLaMA, or Llama3
- **Ensemble OCR**: Combines Tesseract and TrOCR for improved accuracy
- **PDF anonymization**: Creates blackened PDFs with sensitive regions automatically masked
- **Batch processing**: Handles multiple reports with comprehensive error handling

### FrameCleaner
Designed for real-time video frame anonymization featuring:
- **Hardware-accelerated processing**: NVIDIA NVENC support with CPU fallback
- **Streaming video processing**: Processes videos without full re-encoding when possible
- **Adaptive frame sampling**: Optimizes performance for long videos (>10,000 frames)
- **Multiple anonymization strategies**: Frame removal or mask overlay techniques
- **ROI-based masking**: Device-specific region masking for endoscopic equipment

## Default Return Format

LX Anonymizer will return a sensitive meta compliant dict when running either of the main client functions above.

## Highlights
- **End-to-end anonymization** of PDFs and video sequences using OCR, NER, and pseudonymization helpers.
- **Modular pipeline** that lets you choose between Tesseract, TrOCR, ensemble OCR, and multiple metadata extractors.
- **Hardware optimization** with NVENC acceleration for real-time video processing and streaming capabilities.
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

### From Nix

There exists an output from the flake using devenv to package lx-anonnymizer. We recommend using devenv with  this in devenv.yaml

```yaml
inputs:
  lx-anonymizer:
    url: github:wg-lux/lx-anonymizer
```

### From PyPI
```bash
pip install lx-anonymizer
```

Install extras only when you need the corresponding feature set:
```bash
pip install "lx-anonymizer[ocr]"      # TrOCR, tesserocr, CRAFT helpers
pip install "lx-anonymizer[llm]"      # Ollama client helpers
pip install "lx-anonymizer[nlu]"      # Flair NER
pip install "lx-anonymizer[django]"   # Django integration
pip install "lx-anonymizer[dev]"      # local development tooling
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

## Configuration
Settings are loaded from environment variables and an optional `.env` file. See
[`SETTINGS.md`](SETTINGS.md) for a quick overview and example configuration.

## Model downloads
After installation, fetch the German spaCy model used by the report pipeline:
```bash
python -m spacy download de_core_news_lg
```

Also, install the ollama client on your device and ensure it is running with your desired llm for llm support:
```
ollama run deepseek-r1
ollama run llama-3.2
```
Caution: This is only recommended on devices with sufficient gpu capabilities

The EAST detector now downloads on first use, not on import. TrOCR and other optional OCR assets download only when those paths are exercised. For air-gapped deployments, pre-seed the required model files before running the relevant pipeline steps.

## Quickstart

### CLI Usage

#### Image / PDF Pipeline
```bash
# Process a single image or PDF with the packaged console script
lx-anonymizer -i report.pdf

# Use a custom EAST model and device profile
lx-anonymizer -i frame.png -east /models/frozen_east_text_detection.pb -d olympus_cv_1500

# Return validation metadata in addition to the output path
lx-anonymizer -i report.pdf -V
```

**Useful CLI options:**
- `-d/--device` selects the device profile used for ROI handling.
- `-c/--min-confidence`, `-w/--width`, and `-e/--height` tune EAST detection.
- `-V/--validation` returns extra validation metadata.
- `python -m lx_anonymizer.cli --help` shows the same CLI help as `lx-anonymizer --help`.

### Python API

#### ReportReader API
```python
from lx_anonymizer import ReportReader

# Basic usage
reader = ReportReader(locale="de_DE")
original, anonymized, meta, pdf_path = reader.process_report(
    pdf_path="/path/to/report.pdf",
    use_ensemble=True,
    use_llm_extractor="deepseek",
)

# Create anonymized PDF with blackened sensitive regions
original, anonymized, meta, anonymized_pdf = reader.process_report(
    pdf_path="/path/to/report.pdf",
    create_anonymized_pdf=True,
    anonymized_pdf_output_path="/path/to/output.pdf"
)

# Advanced processing with region cropping
original, anonymized, meta, cropped_regions, pdf_path = reader.process_report_with_cropping(
    pdf_path="/path/to/report.pdf",
    crop_output_dir="/path/to/cropped_regions",
    crop_sensitive_regions=True,
    use_llm_extractor="deepseek"
)
```

#### FrameCleaner API
```python
from lx_anonymizer.frame_cleaner import FrameCleaner
from pathlib import Path

# Initialize with hardware acceleration
cleaner = FrameCleaner(use_llm=True)

# Clean video with mask overlay (preserves all frames)
cleaned_path, metadata = cleaner.clean_video(
    video_path=Path("endoscopy.mp4"),
    endoscope_image_roi={"x": 550, "y": 0, "width": 1350, "height": 1080},
    endoscope_data_roi_nested={"patient_info": {"x": 10, "y": 10, "width": 300, "height": 50}},
    technique="mask_overlay"
)

# Remove sensitive frames entirely
cleaned_path, metadata = cleaner.clean_video(
    video_path=Path("endoscopy.mp4"),
    endoscope_image_roi=roi_config,
    endoscope_data_roi_nested=data_roi_config,
    technique="remove_frames"
)
```

See [`tests/test_report_reader_init.py`](tests/test_report_reader_init.py) and [`tests/test_frame_cleaner.py`](tests/test_frame_cleaner.py) for concrete usage patterns.

## Advanced Features

### ReportReader Capabilities
- **Intelligent OCR Fallback**: Automatically switches to OCR when PDF text extraction yields poor results
- **Multi-LLM Support**: DeepSeek, MedLLaMA, and Llama3 integration for enhanced medical entity extraction
- **Ensemble OCR**: Combines multiple OCR engines (Tesseract + TrOCR) for improved accuracy
- **PDF Anonymization**: Creates masked PDFs with sensitive regions automatically blackened
- **Batch Processing**: Processes multiple reports with error recovery and progress tracking
- **Metadata Validation**: Cross-validates extracted information using multiple extraction methods

### FrameCleaner Capabilities
- **Adaptive Sampling**: Automatically samples frames for long videos (>10,000 frames) to optimize performance
- **Hardware Acceleration**: NVIDIA NVENC support with automatic CPU fallback for unsupported systems
- **Streaming Processing**: Uses FFmpeg streaming and named pipes to minimize memory usage and processing time
- **ROI-based Processing**: Device-specific region configurations for endoscopic equipment (Olympus CV-1500, etc.)
- **Multiple Anonymization Strategies**:
  - **Mask Overlay**: Blacks out sensitive regions while preserving video timeline
  - **Frame Removal**: Completely removes sensitive frames from the video stream
- **Quality Optimization**: Automatic pixel format conversion and codec selection for minimal quality loss

### Performance Optimizations
- **Stream Copy Operations**: Avoids re-encoding when possible, using FFmpeg's `-c copy` for maximum speed
- **Named Pipe Support**: In-memory video streaming for frame removal operations
- **Batch Metadata Extraction**: Processes multiple frames simultaneously for improved efficiency
- **Hardware Detection**: Automatically detects and uses available hardware acceleration (NVENC, QuickSync)

## Data directories
By default, outputs live in `~/etc/lx-anonymizer/{data,temp}`. Adjust them in [`lx_anonymizer/directory_setup.py`](lx_anonymizer/directory_setup.py). Clean `temp` regularly to avoid large intermediate artefacts.

## Development workflow
- **Code quality**: `uv run flake8` for linting and formatting
- **Testing**:
  - CPU-friendly tests: `uv run pytest -m "not gpu"`
  - GPU-accelerated tests: `uv run pytest -m gpu` (requires CUDA-capable hardware)
  - Integration tests: `uv run pytest tests/test_cli_integration.py`
  - Frame processing tests: `uv run pytest tests/test_frame_cleaner.py`
- **Performance profiling**: Use `--log-level DEBUG` for detailed timing information
- **Build**: `uv run python -m build` for wheel creation
- **Full validation**: `scripts/run_checks.sh` for comprehensive local testing

## Testing Medical Workflows
- **ReportReader**: Test with sample medical PDFs in German and English
- **FrameCleaner**: Validate with endoscopic video files (MP4, AVI formats supported)
- **Integration**: Use `example_anonymize_pdf.py` for end-to-end testing scenarios

## Project roadmap
1. **Release Management**:
   - Publish the slim base wheel to TestPyPI/PyPI
   - Continue separating optional GPU/LLM workloads behind extras
   - Automate release workflow (wheel + sdist upload, GitHub release notes)
2. **API Enhancement**:
   - Expose REST/gRPC service with validation UI
   - WebSocket support for real-time video processing
   - Enhanced batch processing APIs
3. **Performance & Scalability**:
   - Distributed processing support for large video collections
   - Advanced caching mechanisms for repeated processing
   - Multi-GPU support for FrameCleaner operations
4. **Medical Workflow Integration**:
   - DICOM support for medical imaging workflows
   - HL7 FHIR integration for healthcare systems
   - Advanced medical entity recognition models

## Contributing
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines, testing instructions, and communication channels.

## License
Released under the [MIT License](LICENSE).

## Contact
Questions? Email lux@coloreg.de .

