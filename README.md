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
  - spaCy `de_core_news_sm` model for German NER. Source installs with `uv`
    use the locked model wheel; other runtime environments may need an explicit
    install.
  - Torch vision/audio for video OCR workloads
  - local or remote LLM-backed metadata extraction

## Installation

### From Nix

The repository exposes flake packages and can be consumed directly from another
project's `devenv.yaml`:

```yaml
inputs:
  lx-anonymizer:
    url: github:wg-lux/lx-anonymizer
```

After adding the input, reference the package through your own `devenv.nix` or
flake outputs. You do not need to commit or publish local `result` or
`result-app` symlinks for this to work.

### From PyPI
```bash
pip install lx-anonymizer
```

The base package installs the public API, CLI, PDF/image processing, detector
training, spaCy-based metadata extraction, and PyTesseract fallback OCR. Install
extras only when you need the corresponding hardware or development feature set:
```bash
pip install "lx-anonymizer[dev]"      # local development tooling
pip install "lx-anonymizer[cpu]"      # CPU PyTorch wheel selection
pip install "lx-anonymizer[gpu]"      # CUDA PyTorch wheel selection
```

### From source
```bash
git clone https://github.com/wg-lux/lx-anonymizer.git
cd lx-anonymizer
uv sync --extra dev --extra cpu  # development + CPU PyTorch stack
uv sync --extra gpu  # CUDA 12.8 PyTorch-dependent features
```

The `cpu` and `gpu` extras are mutually exclusive in uv. The CPU extra routes
`torch`, `torchaudio`, and `torchvision` to PyTorch's CPU wheel index; the GPU
extra routes them to PyTorch's CUDA 12.8 wheel index and adds
`onnxruntime-gpu`.

### Nix development shell
```bash
direnv allow
nix develop
```
This loads GPU, OCR, and tooling dependencies declared in `devenv.nix`.

## Packaging

### Python package

PyPI releases now use a split artifact strategy:

- platform wheels are built in GitHub Actions with `maturin` and include the Rust extension
- the source distribution is still built from `pyproject.toml` with `python -m build --sdist`

For a local source-package sanity check:

```bash
uv build --sdist
uv run python scripts/audit_distribution.py dist/*.tar.gz
```

If you build the local wheel on a non-manylinux host (for example Nix), pass the
desired compatibility target explicitly:

```bash
make pypi-wheel PYPI_COMPATIBILITY=manylinux_2_34
```

CI uses `maturin --manylinux auto` via `PyO3/maturin-action` to produce the
published Linux wheels.

The published Python package is the complete baseline install path. Only
hardware-specific PyTorch wheel selection and development/build tools remain in
extras; type stubs and test tools stay outside the runtime dependency set.

### Native extension

The repository also contains an optional Rust extension used for local and Nix
packaging. The Python code loads it opportunistically through
`lx_anonymizer._native` and falls back to pure Python implementations when the
native module is unavailable or only partially implemented.

PyPI wheels built by CI now include this extension. Pure-Python fallback still
exists for environments that install from source without a compiled native
module.

Local prebuilt extension files, generated reports, study data, and cache files
are excluded from PyPI artifacts. CI audits each wheel and sdist before upload.

### Nix packages

The flake exports multiple package variants, including the base CLI package and
a native-enabled package:

```bash
nix build .#lx-anonymizer
nix build .#lx-anonymizer-with-native
```

Those commands create local `./result` symlinks for inspection on your machine.
They are build outputs, not repository contents, and should remain uncommitted.

### Release guidance

- Use `uv build --sdist` and `uv run python scripts/audit_distribution.py dist/*.tar.gz` to validate the source distribution locally.
- Use GitHub Actions to build release wheels with `maturin`.
- Use `nix build .#lx-anonymizer` or `nix build .#lx-anonymizer-with-native` to validate flake packaging.
- Do not commit `result` or `result-app`.
- Configure PyPI trusted publishing before the first tagged release.
- Prefer a TestPyPI dry run before the first production PyPI publication.

### Release workflow

The intended release path is now:

1. Push a branch and let CI build wheels and the sdist.
2. Verify the wheel smoke tests pass on Linux and macOS.
3. Run a TestPyPI publication from the release workflow if this is the first native-wheel release.
4. Tag `vX.Y.Z` to trigger the production publish workflow.

The release workflow publishes:

- native wheels built with `maturin`
- an sdist built with `python -m build --sdist`

## Configuration
Settings are loaded from environment variables and an optional `.env` file. See
[`SETTINGS.md`](SETTINGS.md) for a quick overview and example configuration.

## Model downloads
The default German spaCy model is `de_core_news_sm`. On first use, LX Anonymizer
loads the model if it is installed and otherwise downloads it with the same
Python interpreter that is running the application. To pre-install it, run:
```bash
python -m spacy download de_core_news_sm
```

Clinical/strict deployments fail loudly when the configured model is missing.
Automatic download is enabled by default. Set
`LX_ANONYMIZER_SPACY_AUTO_DOWNLOAD=0` or `SPACY_AUTO_DOWNLOAD=False` to disable
network installation. Outside clinical/strict profiles, disabling it permits
the degraded blank fallback.

Start a compatible LLM server exposing either an OpenAI-compatible API or Ollama:
```bash
# Default local Gemma 4 setup used for OCR and text recognition
bash scripts/provision_ollama_gemma4.sh
# In a devenv shell, the equivalent task is:
devenv tasks run ollama:provision-gemma4

# Alternative high-throughput setup
vllm serve Qwen/Qwen3.5-9B --port 8000
```
Caution: This is only recommended on devices with sufficient gpu capabilities

The EAST detector now downloads on first use, not on import. TrOCR and other optional OCR assets download only when those paths are exercised. For air-gapped deployments, pre-seed the required model files before running the relevant pipeline steps.

## Quickstart

### CLI Usage

Install the package, then use `lx-anonymizer --help` as the single command
index. Every workflow has its own help page:

```bash
# List all image, report, evaluation, dataset, export, and training commands
lx-anonymizer --help

# Show options for one workflow without importing or running the pipeline
lx-anonymizer report --help
lx-anonymizer evaluate-midi-b --help
```

#### Image / PDF Pixel Pipeline

```bash
# Process a single image or PDF
lx-anonymizer image frame.png

# Use a custom EAST model and device profile
lx-anonymizer image frame.png \
  --east /models/frozen_east_text_detection.pb \
  --device olympus_cv_1500

# Return validation metadata in addition to the output path
lx-anonymizer image report.pdf --validation

# The historical spelling remains available
lx-anonymizer -i frame.png
```

#### Report Pipeline

The report command validates the source snapshot and writes one unpublished PDF
candidate to an attempt-owned directory. Its standard output is one JSON object,
which can be consumed directly by shell tooling.

```bash
lx-anonymizer report report.pdf \
  --output-directory ./attempt-output \
  --no-llm

# Optional extraction modes and a caller-supplied attempt identity
lx-anonymizer report report.pdf \
  --output-directory ./attempt-output \
  --attempt-id 12345678-1234-5678-1234-567812345678 \
  --ensemble \
  --llm
```

The output directory may be new or existing, but the generated attempt artifact
must not already exist. A new UUID is generated when `--attempt-id` is omitted.

#### Evaluation, Export, Dataset, and Training Tools

The previously separate console scripts are also available as subcommands:

```bash
lx-anonymizer evaluate-midi-b --help
lx-anonymizer export-dicom --help
lx-anonymizer generate-phi-data --help
lx-anonymizer generate-endoscopy-stickers --help
lx-anonymizer generate-midi-b-phi-data --help
lx-anonymizer generate-radphi-data --help
lx-anonymizer train-phi --help
```

The historical `lx-anonymizer-evaluate-midi-b`,
`lx-anonymizer-export-dicom`, `lx-anonymizer-generate-*`, and
`lx-anonymizer-train-phi` executables remain as compatibility aliases.
`python -m lx_anonymizer.cli` provides the same command interface.

Video import is deliberately not exposed as a standalone production shell
workflow. `endoreg-db` owns durable attempts, leases, encrypted staging,
validation, and publication; it creates one `FrameCleaner` per attempt and uses
the Python API described below.

### Python API

The three central strands use the same shape: construct a typed request, then
call `processor.process(request)` to receive a typed result with an
`artifact_path`. The historical `main(...)`, `clean_video(...)`, and
`process_report(...)` methods remain compatibility wrappers.

#### Image/PDF API

```python
from pathlib import Path

from lx_anonymizer import ImageAnonymizer
from lx_anonymizer.processing_contracts import ImageAnonymizationRequest

attempt_directory = Path("/path/to/image-attempt")
attempt_directory.mkdir(parents=True)
result = ImageAnonymizer().process(
    ImageAnonymizationRequest(
        source_path=Path("/path/to/image.png"),
        output_directory=attempt_directory,
    )
)
print(result.artifact_path, result.metadata)
```

#### ReportReader API
```python
import hashlib
from pathlib import Path
from uuid import uuid4

from lx_anonymizer import ReportReader
from lx_anonymizer.report_contracts import (
    ReportAnonymizationOptions,
    ReportAnonymizationRequest,
)

source_path = Path("/path/to/report.pdf")
source_bytes = source_path.read_bytes()
attempt_directory = Path("/path/to/attempt")
attempt_directory.mkdir(parents=True)

request = ReportAnonymizationRequest(
    attempt_id=uuid4(),
    source_path=source_path,
    source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    source_size_bytes=len(source_bytes),
    output_directory=attempt_directory,
    options=ReportAnonymizationOptions(use_ensemble=True, use_llm=True),
)

reader = ReportReader(locale="de_DE")
result = reader.process(request)
print(result.artifact_path, result.artifact_sha256)

# Advanced processing with region cropping
original, anonymized, meta, cropped_regions, anonymized_pdf = reader.process_report_with_cropping(
    pdf_path="/path/to/report.pdf",
    crop_output_dir="/path/to/cropped_regions",
    crop_sensitive_regions=True,
    use_llm=True
)
```

`ReportReader` is the canonical report-oriented entry point for immutable PDF
snapshots.

`ReportReader(...)` constructor:
- `report_root_path`: optional base path for report assets.
- `locale`: Faker locale for pseudonymized replacements.
- `employee_first_names` / `employee_last_names`: optional replacement pools.
- `flags`: optional parsing markers merged with `DEFAULT_SETTINGS["flags"]`.
- `text_date_format`: output format used for anonymized date text.

`process(...)` (and its compatibility alias `process_report(...)`) accepts one strictly validated
`ReportAnonymizationRequest`. The caller supplies the immutable source identity,
an attempt-owned output directory, and processing options. The method always
creates and validates an anonymized PDF without choosing a canonical publication
path.

It returns a frozen `ReportAnonymizationResult` containing original and
anonymized text, typed sensitive metadata, the attempt-local artifact path,
artifact size and SHA-256, structural PDF validation, and anonymizer provenance.

`process_report_with_cropping(...)` is a separate diagnostic helper with:
- `crop_output_dir`: where cropped sensitive regions are written.
- `crop_sensitive_regions`: enable or disable crop extraction.
- `anonymization_output_dir`: output directory for the crop-based anonymized PDF.

`process_report_with_cropping(...)` returns:
- `original_text`
- `anonymized_text`
- `report_meta`
- `cropped_regions_info`: mapping of cropped sensitive regions.
- `anonymized_pdf_path`: `Path | None`

#### FrameCleaner API
```python
from fractions import Fraction
from pathlib import Path

from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.processing_contracts import VideoAnonymizationRequest

result = FrameCleaner(use_llm=True).process(
    VideoAnonymizationRequest(
        source_path=Path("endoscopy.mp4"),
        output_path=Path("attempt/candidate.mp4"),
        source_frame_rate=Fraction(25, 1),
        endoscope_image_roi={"x": 550, "y": 0, "width": 1350, "height": 1080},
        endoscope_data_roi_nested={
            "patient_info": {"x": 10, "y": 10, "width": 300, "height": 50}
        },
        technique="mask_overlay",
    )
)
print(result.artifact_path, result.metadata)
```

`FrameCleaner` is the video-oriented entry point for endoscopy footage and
frame-level overlays.

`FrameCleaner(...)` constructor:
- `use_llm`: enables provider-backed batch metadata enrichment when available.
- `use_minicpm` and `minicpm_config`: reserved for optional OCR backends.

`clean_video(...)` parameters:
- `video_path`: input video file.
- `endoscope_image_roi`: flat ROI dict for the visible endoscope image, typically with `x`, `y`, `width`, `height`.
- `endoscope_data_roi_nested`: nested ROI mapping for text-bearing overlay regions such as patient info blocks.
- `output_path`: optional explicit output path.
- `technique`: one of `mask_overlay`, `remove_frames`, or `extract_only`.
- `device`: device profile name, defaulting to `olympus_cv_1500`.

`clean_video(...)` behavior by technique:
- `mask_overlay`: preserves the timeline and overlays masks onto sensitive regions.
- `remove_frames`: drops sensitive frames and rewrites the stream.
- `extract_only`: does metadata extraction without producing a masked/removal-focused anonymization pass.

`clean_video(...)` returns:
- `output_video_path`: resulting video path. With `extract_only`, this is still the path chosen for the run.
- `sensitive_meta`: accumulated metadata dictionary extracted from sampled frames.

#### Retraining and using the region detector

Training is included in the standard installation. Its typed result creates the
checksum-pinned runtime configuration accepted by every strand:

```python
from pathlib import Path

from lx_anonymizer import ImageAnonymizer, ReportReader
from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.text_detection.phi_region_detector import CustomPhiRegionDetector
from lx_anonymizer.text_detection.phi_region_detector_training import (
    PhiRegionDetectorTrainingConfig,
    train_phi_region_detector,
)

training = train_phi_region_detector(
    PhiRegionDetectorTrainingConfig(
        dataset_yaml=Path("datasets/phi/data.yaml"),
        output_dir=Path("runs/phi"),
    )
)
detector = CustomPhiRegionDetector(training.detector_config(required=True))

image_processor = ImageAnonymizer(region_detector=detector)
video_processor = FrameCleaner(region_detector=detector)
report_processor = ReportReader(region_detector=detector)
```

ROI guidance:
- Use `endoscope_image_roi` for the main picture area that may need masking.
- Use `endoscope_data_roi_nested` for device-specific overlay fields.
- The helper stack normalizes common ROI key variants, but using `x`, `y`, `width`, `height` directly is the least ambiguous form.

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
By default, outputs live in `~/etc/lx-anonymizer/{data,temp}`. Adjust them in
[`lx_anonymizer/setup/directory_setup.py`](lx_anonymizer/setup/directory_setup.py).
Clean `temp` regularly to avoid large intermediate artefacts.

## Development workflow
- **Code quality**: `uv run flake8` for linting and formatting
- **Testing**:
  - CPU-friendly tests: `uv run pytest -m "not gpu"`
  - GPU-accelerated tests: `uv run pytest -m gpu` (requires CUDA-capable hardware)
  - Integration tests: `uv run pytest tests/test_cli_integration.py`
  - Frame processing tests: `uv run pytest tests/test_frame_cleaner.py`
- **Performance profiling**: Use `--log-level DEBUG` for detailed timing information
- **Build**: `uv run python -m build --sdist` for local sdist validation; GitHub Actions builds release wheels
- **Full validation**: `scripts/run_checks.sh` for comprehensive local testing

## Testing Medical Workflows
- **ReportReader**: Test with sample medical PDFs in German and English
- **FrameCleaner**: Validate with endoscopic video files (MP4, AVI formats supported)
- **Integration**: Use `example_anonymize_pdf.py` for end-to-end testing scenarios

## Project roadmap
1. **Release Management**:
   - Continue hardening native-wheel publishing across release targets
   - Continue separating optional GPU/LLM workloads behind extras
   - Extend release automation with GitHub release notes and TestPyPI promotion flow
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
