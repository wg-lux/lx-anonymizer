# Settings

LX Anonymizer reads configuration from environment variables and an optional `.env`
file in the repository root. The settings are defined in `lx_anonymizer/config.py`
and exposed as a singleton `settings` object.

The provider-agnostic LLM modules live under `lx_anonymizer.llm.llm_*`.

You can import them in code:

```python
from lx_anonymizer.config import settings

if settings.LLM_ENABLED:
    ...
```

## Common knobs

- `MODE` (default: `production`) - general mode flag.
- `DEBUG_SAVE_FRAMES` (default: `False`) - save intermediate frames for debugging.
- `SPACY_MODEL` / `LX_ANONYMIZER_SPACY_MODEL` (default: `de_core_news_sm`) -
  German spaCy model used by the metadata extractors.
- `SPACY_AUTO_DOWNLOAD` / `LX_ANONYMIZER_SPACY_AUTO_DOWNLOAD` (default: `False`)
  - allow the extractor to download the configured spaCy model if it is missing.
- `SPACY_STRICT` / `LX_ANONYMIZER_SPACY_STRICT` (default: `False`) - fail
  instead of using the degraded blank fallback when the configured spaCy model
  is missing. Clinical profiles are strict even when this is unset.
- `LLM_ENABLED` (default: `True`) - enable local Ollama-backed OCR, text
  recognition, and metadata extraction.
- `LLM_PROVIDER` (default: `ollama`) - provider backend, one of `vllm` or `ollama`.
- `LLM_BASE_URL` (default: empty) - optional explicit endpoint override. If unset,
  the code uses `http://127.0.0.1:8000` for `vllm` and `http://127.0.0.1:11434`
  for `ollama`.
- `LLM_MODEL` (default: `lx-gemma4-e2b-json`) - provisioned Gemma 4 OCR and JSON
  model name.
- `LLM_TIMEOUT` (default: `120`) - request timeout in seconds.
- `OLLAMA_OCR_ENABLED` (default: `True`) - send report images and high-quality
  sampled video frames to Gemma 4 vision OCR. Conventional OCR remains the
  candidate source and failure fallback.
- `OLLAMA_OCR_CONFIDENCE` (default: `0.5`) - explicit proxy used only when Gemma
  recognizes text and the conventional OCR candidate was empty; generative
  vision responses do not expose a calibrated OCR confidence.
- `MAX_FRAMES_TO_SAMPLE` (default: `24`) - cap for OCR frames per video.
- `SMART_EARLY_STOPPING` (default: `True`) - stop early when metadata is complete
  in extraction-only mode.
- `OCR_CONFIDENCE_THRESHOLD` (default: `0.6`) - OCR confidence gate.
- `PHI_REGION_DETECTOR_MODEL_PATH` (default: empty) - optional local ONNX model
  path for a custom retrainable PHI-region detector. When set, its regions are
  added to the normal EAST/OCR detections.
- `PHI_REGION_DETECTOR_MODEL_SHA256` (default: empty) - optional SHA-256 pin for
  the local detector artifact.
- `PHI_REGION_DETECTOR_REQUIRED` (default: `False`) - fail the anonymization call
  if the configured custom detector cannot run.
- `PHI_REGION_DETECTOR_CONFIDENCE` (default: `0.35`) - minimum model confidence.
- `PHI_REGION_DETECTOR_NMS_THRESHOLD` (default: `0.45`) - box overlap threshold
  for non-maximum suppression.
- `PHI_REGION_DETECTOR_INPUT_SIZE` (default: `640`) - square model input size.
- `PHI_REGION_DETECTOR_RESIZE_MODE` (default: `letterbox`) - preserve aspect
  ratio with Ultralytics-compatible padding; `stretch` is available only for
  legacy models trained with distorted square inputs.
- `PHI_REGION_DETECTOR_BOX_FORMAT` (default: `yolo_xywh`) - `yolo_xywh` or `xyxy`.
- `PHI_REGION_DETECTOR_SCORE_FORMAT` (default: `class_scores`) -
  `class_scores` for YOLOv8-style outputs or `objectness` for YOLOv5-style
  outputs.
- `PHI_REGION_DETECTOR_CLASS_IDS` (default: empty) - optional comma-separated
  class ids to keep.
- `MASKING_STRATEGY` (default: `mask_overlay`) - `mask_overlay` or `remove_frames`.
- `VIDEO_ENCODER` (default: `auto`) - `auto`, `h264_nvenc`, or `libx264`.

## Example `.env`

```ini
MODE=production
SPACY_MODEL=de_core_news_sm
SPACY_AUTO_DOWNLOAD=False
SPACY_STRICT=False
LLM_ENABLED=True
LLM_PROVIDER=ollama
LLM_BASE_URL=
LLM_MODEL=lx-gemma4-e2b-json
LLM_TIMEOUT=120
LLM_MAX_CALLS_PER_VIDEO=1
LLM_MIN_TEXT_LENGTH=32
OLLAMA_OCR_ENABLED=True
OLLAMA_OCR_CONFIDENCE=0.5
REPORT_LLM_MIN_TEXT_LENGTH=64
REPORT_OCR_CORRECTION_MIN_TEXT_LENGTH=120
MAX_FRAMES_TO_SAMPLE=24
SMART_EARLY_STOPPING=True
OCR_CONFIDENCE_THRESHOLD=0.6
PHI_REGION_DETECTOR_MODEL_PATH=/home/admin/lx-anonymizer/study_data/phi_model_runs/phi-yolov8n-radphi-synthetic-stickers-midi-refine-seed0-960-20260715/weights/best.onnx
PHI_REGION_DETECTOR_MODEL_SHA256=61c2b58e283733c391c577df36dce057d7424f5d611d27ae0867e30ff684a5bd
PHI_REGION_DETECTOR_REQUIRED=True
PHI_REGION_DETECTOR_CONFIDENCE=0.05
PHI_REGION_DETECTOR_NMS_THRESHOLD=0.45
PHI_REGION_DETECTOR_INPUT_SIZE=960
PHI_REGION_DETECTOR_RESIZE_MODE=letterbox
PHI_REGION_DETECTOR_BOX_FORMAT=yolo_xywh
PHI_REGION_DETECTOR_SCORE_FORMAT=class_scores
PHI_REGION_DETECTOR_CLASS_IDS=0
MASKING_STRATEGY=mask_overlay
VIDEO_ENCODER=auto
```

## Notes

- `MAX_FRAMES_TO_SAMPLE` is the main performance lever for long videos.
- `LLM_BASE_URL` is optional. Leave it empty unless you want to point at a
  non-default host or port.
- Gemma 4 OCR runs through the local Ollama endpoint. Conventional OCR remains
  active as both an input candidate and a failure fallback.
- The shipped defaults are tuned for smooth laptop usage, not maximum throughput.
  For server workloads, switch `LLM_PROVIDER=vllm` and increase sampling/call budgets.
- `SMART_EARLY_STOPPING` only applies when `FrameCleaner.clean_video()` is called
  with `technique="extract_only"` so masking/removal still process the full video.
- The custom PHI-region detector is local-only and additive. It never downloads
  artifacts and does not replace the deterministic EAST/OCR redaction path.
- The release profile pins the detector by SHA-256 and sets
  `PHI_REGION_DETECTOR_REQUIRED=True`, so a missing, corrupt, or unreadable model
  fails the anonymization call instead of silently continuing without the learned
  detector. Deployments at a different filesystem location must change the path
  while preserving the pinned digest and inference settings.
- OCR remains a cascade: RapidOCR is preferred, TesseOCR/PyTesseract are local
  fallbacks, and Gemma vision OCR may refine high-quality samples. The configured
  `OLLAMA_OCR_CONFIDENCE` is a proxy when conventional OCR finds no candidate; it
  is not a calibrated probability. Human review remains required before release
  or secondary use of anonymized data.
