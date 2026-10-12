# Model loading and fallback

The report and video entry points load models only when needed. Each ordered
`ModelBlock` contains a name, a spawn-compatible loader, a deadline and an optional
result validator. `ModelChain` uses one worker process at a time and reuses the
selected model between requests. Reordering the blocks changes priority without
changing the runner. Settings and examples are in
[llm-configuration.yml](llm-configuration.yml); implementation evidence is tracked
in [LLMManagement.yml](../../feature-tracking/LLMManagement.yml) and
[FrameCleanerIntegration.yml](../../feature-tracking/FrameCleanerIntegration.yml).

Set environment arrays to change order, for example:

```sh
OCR_MODEL_ORDER='["tesserocr", "rapidocr", "pytesseract"]'
METADATA_MODEL_ORDER='["spacy", "llm"]'
MODEL_BLOCK_TIMEOUT=20
```

The default OCR order is RapidOCR, Tesserocr, Pytesseract. Report image OCR tries
the configured vision model, the existing OCR ensemble, then Tesseract. Vision
is now a first-choice block, rather than an unconditional refinement after
conventional OCR. Reports try the configured LLM and then full local spaCy/regex
extraction. Video metadata retains its per-frame local extraction and bounded
video-level LLM budget; it does not call a language model for every frame.
The direct `FrameCleaner.extract_metadata` method follows metadata block order.

`LLMFactory.create_metadata_extractor()` now configures the provider without
probing it. Its `current_model` represents the configured candidate until first
use, not confirmed availability. The worker performs discovery and honors the
exact configured model and transport policy. `isolated=False` is available for
the low-level provider adapter and tests; it is synchronous and relies on its
caller's execution boundary. The raw OCR/model-service functions likewise remain
low-level primitives; use the bounded facades for production orchestration.

## Deadlines and ownership

`run()` is synchronous but bounded. `await arun()` waits without blocking the
event loop and propagates cancellation after stopping its worker. Model imports,
loading, inference and socket transfers share the block deadline. Failed or hung
workers and their process groups are killed before fallback; the direct worker
is reaped with a one-second limit. Cleanup failure raises instead of starting
another backend. There is no abandoned inference thread or background download.

Workers use Linux process isolation and private sockets; no patient payloads are
written to an intermediate model-runtime file. The socket protocol uses pickle
only for trusted, in-process Python contracts, never external/network data.
Messages are capped at 128 MiB. Factories must be importable, not lambdas or
closures. Launch scripts using these facades need the ordinary multiprocessing
`if __name__ == "__main__":` guard. Daemon multiprocessing workers that forbid
child processes cannot provide this isolation; use a host worker that permits
owned child processes. Host admission, operating-system resource limits and
abrupt host termination remain the host's responsibility.

One chain rejects simultaneous calls. A failing block remains disabled for that
attempt. `reset()` explicitly starts a fresh attempt; `close()` or the context
manager stops workers. Video and canonical report workflows close workers on
both success and failure. The generic runner also installs a finalizer for
standalone use. Recent attempt history is bounded; aggregate usage counters are
per model. Patient metadata is reset between requests even when weights are reused.

## Failure semantics and provenance

Loading errors, inference errors, deadlines and worker crashes advance to the
next block. Invalid recognized sensitive metadata remains a hard error. A valid
empty frame OCR result is distinct from unavailable OCR. Empty LLM metadata
advances to local extraction; empty report OCR advances to the next OCR block.
The final absence of report OCR raises instead of reporting a successful blank
document. Model logs identify the block and outcome without copying exception
values or patient input.

Exhaustive video uses the same full-frame cascade. If every OCR block fails at
any inspected frame, the configured mask is applied to the entire candidate, audio and container
metadata are removed, and the source dimensions and frame sequence are retained.
This applies to `mask_overlay` and `remove_frames`; `extract_only` still produces
no media. A no-model candidate is for reprocessing, not clinical release.

`VideoAnonymizationResult.model_execution` and the dictionary result's
`model_execution` contain `models_used`, `failed_models`, `configured_mask_fallback`, and
`reprocessing_required`. Metadata coverage records `complete=false` and
`cap_reason=model_failure`. Decoding, encoding, source validation and cleanup
errors still raise and remove the owned partial candidate. See the
[video concurrency contract](../VIDEO_IMPORT_CONCURRENCY_CONTRACT.md).

The host must preserve and present the new summary before deployment. The
compute library returns an unpublished candidate; it does not publish, schedule
reprocessing, or replace a validated master. Clinical quality and host review
integration require separate deployment verification.

## Shared masks and host integration

`lx_dtypes.models.interface.video_masks` loads `VideoMaskProfile` definitions
from `lx_dtypes/data/video_masks/*.yml`. Each profile uses the existing
`VideoMaskConfig` coordinate contract. Missing names and invalid files raise;
loading never creates or repairs package files. User-authored masks are supplied
through an explicit runtime root, not by changing the immutable package root.

Endoreg-db serves authenticated `GET`/`POST media/videos/masks/`. POST accepts
`name`, `label`, `device_name` and `mask` (`x`, `y`, `width`, `height`, `image_width`,
`image_height`). Coordinates describe the rectangle to retain; pixels outside
it are masked. Creation is atomic and duplicate names return 409. Runtime
profiles live under `get_runtime_paths().terminology / "video_masks"`.

Select a profile through the existing video anonymization correction endpoint
with `region: {mode: "profile", mask_name: "room_2"}`. The host resolves the
profile before queuing and persists its full validated snapshot. Workers pass
that snapshot to lx-anonymizer without looking up mutable configuration.
The lx-annotate correction form loads this catalogue and can save and select
custom coordinates, including the reference image dimensions and device name.
The frame editor pauses the original video and takes a browser-memory snapshot.
Drag the endoscope rectangle on that still frame; reversed and out-of-bounds
drags are normalized to source pixels. Only coordinates are sent to the API,
never the captured frame. Switching video or preview clears the still. Human review
remains required. Ship the matching lx-dtypes, endoreg-db and lx-anonymizer
revisions together; an older lx-dtypes package does not contain this loader.

Model factories must invoke local implementations. Starting another isolated
`ModelChain` from inside a model worker is rejected before spawning, so nested
adapters cannot create independent process groups outside the attempt deadline.
