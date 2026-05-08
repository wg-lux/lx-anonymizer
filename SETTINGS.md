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
- `LLM_ENABLED` (default: `False`) - enable LLM metadata extraction. The package
  defaults to off so it behaves conservatively when imported by other repos.
- `LLM_PROVIDER` (default: `ollama`) - provider backend, one of `vllm` or `ollama`.
- `LLM_BASE_URL` (default: empty) - optional explicit endpoint override. If unset,
  the code uses `http://127.0.0.1:8000` for `vllm` and `http://127.0.0.1:11434`
  for `ollama`.
- `LLM_MODEL` (default: `qwen2.5:7b-instruct`) - preferred model name for the selected
  provider.
- `LLM_TIMEOUT` (default: `45`) - request timeout in seconds.
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
LLM_ENABLED=True
LLM_PROVIDER=ollama
LLM_BASE_URL=
LLM_MODEL=qwen2.5:7b-instruct
LLM_TIMEOUT=45
LLM_MAX_CALLS_PER_VIDEO=1
LLM_MIN_TEXT_LENGTH=32
REPORT_LLM_MIN_TEXT_LENGTH=64
REPORT_OCR_CORRECTION_MIN_TEXT_LENGTH=120
MAX_FRAMES_TO_SAMPLE=24
SMART_EARLY_STOPPING=True
OCR_CONFIDENCE_THRESHOLD=0.6
PHI_REGION_DETECTOR_MODEL_PATH=
PHI_REGION_DETECTOR_MODEL_SHA256=
PHI_REGION_DETECTOR_REQUIRED=False
PHI_REGION_DETECTOR_CONFIDENCE=0.35
PHI_REGION_DETECTOR_NMS_THRESHOLD=0.45
PHI_REGION_DETECTOR_INPUT_SIZE=640
PHI_REGION_DETECTOR_BOX_FORMAT=yolo_xywh
PHI_REGION_DETECTOR_SCORE_FORMAT=class_scores
PHI_REGION_DETECTOR_CLASS_IDS=
MASKING_STRATEGY=mask_overlay
VIDEO_ENCODER=auto
```

## Notes

- `MAX_FRAMES_TO_SAMPLE` is the main performance lever for long videos.
- `LLM_BASE_URL` is optional. Leave it empty unless you want to point at a
  non-default host or port.
- `LLM_ENABLED=False` is the conservative default for library consumers. Turn it on
  explicitly in the calling repo or deployment environment.
- The shipped defaults are tuned for smooth laptop usage, not maximum throughput.
  For server workloads, switch `LLM_PROVIDER=vllm` and increase sampling/call budgets.
- `SMART_EARLY_STOPPING` only applies when `FrameCleaner.clean_video()` is called
  with `technique="extract_only"` so masking/removal still process the full video.
- The custom PHI-region detector is local-only and additive. It never downloads
  artifacts and does not replace the deterministic EAST/OCR redaction path.
