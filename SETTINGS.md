# Settings

LX Anonymizer reads configuration from environment variables and an optional `.env`
file in the repository root. The settings are defined in `lx_anonymizer/config.py`
and exposed as a singleton `settings` object.

You can import them in code:

```python
from lx_anonymizer.config import settings

if settings.LLM_ENABLED:
    ...
```

## Common knobs

- `MODE` (default: `production`) - general mode flag.
- `DEBUG_SAVE_FRAMES` (default: `False`) - save intermediate frames for debugging.
- `LLM_ENABLED` (default: `True`) - enable LLM metadata extraction.
- `LLM_MODEL` (default: `llama3.2:1b`) - preferred Ollama model name.
- `LLM_TIMEOUT` (default: `30`) - request timeout in seconds.
- `MAX_FRAMES_TO_SAMPLE` (default: `50`) - cap for OCR frames per video.
- `SMART_EARLY_STOPPING` (default: `True`) - stop early when metadata is complete
  in extraction-only mode.
- `OCR_CONFIDENCE_THRESHOLD` (default: `0.6`) - OCR confidence gate.
- `MASKING_STRATEGY` (default: `mask_overlay`) - `mask_overlay` or `remove_frames`.
- `VIDEO_ENCODER` (default: `auto`) - `auto`, `h264_nvenc`, or `libx264`.

## Example `.env`

```ini
MODE=production
LLM_ENABLED=True
LLM_MODEL=llama3.2:1b
LLM_TIMEOUT=15
MAX_FRAMES_TO_SAMPLE=40
SMART_EARLY_STOPPING=True
OCR_CONFIDENCE_THRESHOLD=0.6
MASKING_STRATEGY=mask_overlay
VIDEO_ENCODER=auto
```

## Notes

- `MAX_FRAMES_TO_SAMPLE` is the main performance lever for long videos.
- `SMART_EARLY_STOPPING` only applies when `FrameCleaner.clean_video()` is called
  with `technique="extract_only"` so masking/removal still process the full video.
