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
