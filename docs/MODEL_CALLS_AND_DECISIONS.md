# Model calls and decision flow

This document describes the model-backed calls and the decisions around them for
the two main entrypoints:

- video and frame processing: `lx_anonymizer/frame_cleaner.py`
- report processing: `lx_anonymizer/report_reader.py`

It follows the code that is reachable from `FrameCleaner.clean_video()` and
`ReportReader.process_report()`, including optional report PDF redaction. It also
notes model-related code that looks relevant but is not currently called by those
entrypoints. The description reflects the repository as of 2026-07-15.

## Terminology

"Model call" below includes learned OCR, NLP, object/text detection, and LLM
inference. Regexes, thresholds, sorting, merging, and validation are described as
decisions rather than model calls.

The principal runtime model families are:

| Family | Default/configured model | Purpose |
| --- | --- | --- |
| RapidOCR | PP-OCRv4 mobile detection/classification plus Latin recognition via ONNX Runtime | Primary video-frame OCR when installed |
| Tesseract | Tesseract OCR (`deu+eng` for frame fallback) | Frame and report OCR; OCR on report redaction boxes |
| TrOCR | `microsoft/trocr-base-str` | Optional member of report ensemble OCR |
| spaCy | `de_core_news_sm` by default | Tokenization and rule-based matchers for patient/examiner/endoscope metadata |
| Provider LLM | `lx-gemma4-e2b-json` by default; Ollama or vLLM | Structured metadata extraction and report OCR correction |
| Ollama vision model | The same `LLM_MODEL` setting | Optional image transcription for frames and report pages |
| EAST | `frozen_east_text_detection.pb` | Text-region localization during optional report PDF redaction |
| Custom PHI detector | Configured ONNX model, disabled by default | Additive PHI-region proposals for frames and report redaction |

Configuration defaults are defined in `lx_anonymizer/config.py:13-66`.

## Shared provider LLM path

Both entrypoints construct their structured metadata extractor through
`LLMFactory.create_metadata_extractor()` (`lx_anonymizer/llm/factory.py:5-14`).
The factory passes `LLM_PROVIDER`, the resolved base URL, `LLM_MODEL`, and
`LLM_TIMEOUT` to `LLMMetadataExtractor`.

### Availability and model selection

Construction is itself externally observable, even before inference:

1. `LLMMetadataExtractor` calls the provider's model-list endpoint: Ollama
   `/api/tags` or vLLM `/v1/models` (`llm/llm_extractor.py:291-322`).
2. It uses the configured preferred model if that exact name is available.
3. Otherwise it tries, in priority order:
   `lx-gemma4-e2b-json`, `gemma4:e2b`, then `llama3.2:1b`
   (`llm/llm_extractor.py:155-183,324-375`).
4. If no compatible model is listed, `current_model` remains `None` and the
   entrypoint disables provider-backed metadata extraction.

The model-list check has a five-second request timeout. On an exception it makes
one delayed retry after one second.

### Structured metadata inference

An actual `extract_metadata(text)` call performs the following decisions
(`llm/llm_extractor.py:582-715`):

1. Return a cached result for the exact input text when available. Cache keys are
   truncated MD5 hashes and the in-memory FIFO cache holds 100 entries.
2. Build a prompt using only the first 6,000 characters. The prompt requires
   JSON-only output, forbids invented values, distinguishes DOB from examination
   date, and requests `first_name`, `last_name`, `dob`, `casenumber`, and
   `examination_date` (`llm/llm_extractor.py:377-402`).
3. Call Ollama `/api/chat` with JSON format, temperature 0, and an 8,192-token
   context, or the OpenAI-compatible `/v1/chat/completions` endpoint with JSON
   response format and temperature 0 (`llm/llm_extractor.py:505-552`).
4. Retry an individual HTTP request up to twice with a one-second wait.
5. Parse the response after removing reasoning/Markdown wrappers, validate it as
   `LLMMetadataPayload`, and merge it through `SensitiveMeta`.
6. On timeout, request failure, or invalid JSON, try the next available configured
   model. Return `None` after all models fail.

This means one logical metadata extraction can produce more than one network
inference call because of HTTP retries and model fallback.

### State lifetime caveat

The provider extractor owns a persistent `SensitiveMeta`, and its cached values
are merged back into that same object (`llm/llm_extractor.py:186-235,600-674`).
The spaCy/rule extractor instances also own persistent metadata. Resetting a
`FrameCleaner` video run or a `ReportReader.process_report()` call resets the
entrypoint-level `SensitiveMeta`, but does not recreate those nested extractor
objects. Reusing one entrypoint instance across multiple videos or reports can
therefore allow previously filled, nonblank fields to influence a later result or
the decision to skip an LLM call. This is current state behavior, not an explicit
per-document isolation boundary.

## `FrameCleaner` video pipeline

### High-level flow

```text
construct FrameCleaner
  -> choose OCR backend and optionally initialize provider LLM
  -> sample video frames
  -> for every sampled frame:
       conventional OCR
       -> optional Ollama vision OCR
       -> local spaCy/rule metadata extraction
       -> optional custom PHI detector
       -> sensitive/not-sensitive decision
  -> optional one- or two-candidate batch LLM metadata extraction
  -> remove sensitive sampled frames, mask a fixed overlay ROI, or extract only
  -> emit SensitiveMeta-compatible metadata and provenance
```

### Initialization decisions

`FrameCleaner.__init__()` initializes `FrameOCR`, `FrameMetadataExtractor`,
`PatientDataExtractor`, and `ROIProcessor` (`frame_cleaner.py:138-180`).

Provider metadata extraction is enabled when the constructor's `use_llm` is
true, or when it is omitted and `LLM_ENABLED` is true. It is disabled if factory
construction fails or no current provider model exists
(`frame_cleaner.py:187-229`). The default configuration is enabled.

The `use_minicpm` and `minicpm_config` constructor arguments are stored but not
read anywhere else in `FrameCleaner`; they do not cause a model call.

### Frame sampling decisions

The quality profile controls the maximum number of frames and the `high_quality`
OCR flag (`frame_cleaner.py:68-112`):

| Profile | Maximum sampled frames | High-quality OCR |
| --- | ---: | --- |
| `fast` | min(configured maximum, 12) | no |
| `balanced` | configured maximum, 24 by default | yes |
| `quality` | max(configured maximum, 48) | yes |

Tests impose an additional 12-frame cap. Actual frame iteration chooses a stride
of `max(5, min(ceil(total_frames / target_samples), fps * 2))` and stops after the
target number has been processed (`frame_cleaner_video.py:400-437` and
`frame_cleaner.py:409-490`). Frames are converted to grayscale and histogram
equalized before model inference.

### Call FC-1: frame OCR cascade

Every sampled frame calls `FrameOCR.extract_text_from_frame()`
(`frame_cleaner.py:990-1011`; `ocr/ocr_frame.py:151-217`). The conventional
backend cascade is:

1. **RapidOCR**, if importable. It lazily initializes PP-OCRv4 mobile detection
   and classification plus Latin recognition. ONNX Runtime uses CUDA when
   available under `auto`, otherwise CPU (`ocr/ocr_frame.py:265-387`). Each valid
   configured ROI is processed independently; otherwise the whole frame is
   processed. Detected lines are sorted top-to-bottom/left-to-right, joined, and
   their normalized confidences averaged (`ocr/ocr_frame.py:405-567`).
2. **tesserocr**, if RapidOCR is unavailable or fails and tesserocr can initialize.
   It is configured for German (`ocr/ocr_frame.py:389-402`).
3. **pytesseract**, if the earlier backends are unavailable or fail. The frame is
   cropped to a valid flat ROI, contrast-enhanced and sharpened, then processed
   with `deu+eng`, OEM 3, PSM 6, and DPI 300. Only words with confidence greater
   than zero are retained (`ocr/ocr_frame.py:611-647,649-691`).

Nested `endoscope_data_roi_nested` wins over `endoscope_image_roi`; invalid or
absent ROIs result in full-frame OCR (`frame_cleaner.py:885-900`). RapidOCR
currently ignores the `high_quality` value. The flag still gates vision OCR and
is passed to the other OCR backends.

#### Optional FC-1b: Ollama vision OCR

After conventional OCR, one Ollama vision inference is attempted per high-quality
sampled frame only when all of these are true:

- global `LLM_ENABLED` is true;
- `OLLAMA_OCR_ENABLED` is true;
- `LLM_PROVIDER == "ollama"`; and
- the call uses high-quality OCR.

The cropped image (only when exactly one ROI was resolved) and conventional OCR
candidate are sent to `/api/chat`. The prompt demands a literal transcription and
uses the image as source of truth (`ocr/ocr_frame.py:219-263`;
`llm/llm_service.py:138-190`). `[NO_TEXT]` becomes an empty result. Empty or
failed vision output retains conventional OCR.

When vision returns text, confidence still comes from conventional OCR. If the
conventional candidate was empty, the configured proxy confidence (0.5 by
default) is used.

Important control distinction: this call checks global settings, not
`FrameCleaner.use_llm`. Therefore `FrameCleaner(use_llm=False)` disables
structured metadata LLM calls but does **not** disable frame vision OCR when the
global vision conditions remain enabled.

### Call FC-2: local spaCy/rule metadata extraction

For non-empty OCR text, `_unified_metadata_extract()` first calls
`PatientDataExtractor` (`frame_cleaner.py:687-717`). `PatientDataExtractor` runs
the configured spaCy pipeline and a spaCy `Matcher`; extraction itself is mostly
rule patterns rather than generic named-entity labels
(`ner/spacy_extractor.py:353-537`).

The spaCy model is loaded once. The default is `de_core_news_sm`. If it is absent,
the code may explicitly download it when auto-download is enabled, fail loudly in
strict/clinical mode, or use a degraded blank German pipeline otherwise
(`ner/spacy_extractor.py:95-279`).

Decision after the patient extractor:

- If any recognized metadata field has signal, merge it into the accumulated
  `SensitiveMeta`.
- If it has no signal or throws, use `FrameMetadataExtractor`, which applies
  frame-specific regexes for names, DOB, case number, examination date/time,
  examiner, and gender (`ner/frame_metadata_extractor.py:70-129`).

No provider LLM is called at this per-frame metadata stage on the current
`clean_video()` path.

### Call FC-3: optional custom PHI-region detector

Every sampled frame also calls `detect_phi_regions_from_settings()`
(`frame_cleaner.py:902-934`). With the default empty model path this immediately
returns no regions, so no inference occurs.

When configured, it loads an OpenCV-DNN-compatible ONNX model, letterboxes or
stretches the frame to 640×640 by default, performs one forward pass, filters by
confidence (0.35 default) and allowed class IDs, converts coordinates back to the
source image, and applies NMS at 0.45
(`text_detection/phi_region_detector.py:55-199,206-245`). The model is cached.

Configuration and runtime failures are logged and treated as no regions unless
`PHI_REGION_DETECTOR_REQUIRED` is true, in which case they raise.

### Per-frame sensitivity and accumulation decisions

A frame is sensitive when either condition is true (`frame_cleaner.py:1044-1064`):

- local metadata contains a nonblank first name, last name, case number, DOB, or
  gender; or
- the custom PHI detector returned at least one region.

Examination date/time, examiner information, or OCR text alone do not make the
frame sensitive.

Across sampled frames:

- metadata is fill-only merged through `SensitiveMeta`;
- the accumulated metadata is copied into each current frame result, so after a
  sensitive identifier is found, later sampled frames can be classified as
  sensitive from that accumulated identifier even when their own OCR did not
  independently contain it;
- DOB and examination date are reordered when both parse and appear reversed;
- the representative OCR text is the highest-confidence result, with longer text
  breaking confidence ties;
- a sensitive sampled frame index is recorded for `remove_frames`;
- analysis stops early only for `extract_only` and `mask_overlay`, only when smart
  early stopping is enabled, and only when metadata is "complete."

Completeness means one of: full name + DOB; full name + case number + examination
date; or partial name + DOB + case number
(`ner/frame_metadata_extractor.py:147-173`).

### Call FC-4: optional video-level LLM metadata extraction

After frame sampling, `_maybe_enrich_video_metadata()` can make structured LLM
calls (`frame_cleaner.py:543-578,1198-1299`). It runs only when:

- provider LLM use is enabled and a current model exists;
- at least one frame with OCR text was collected;
- local accumulated metadata is not complete; and
- the per-video call budget is not exhausted (`LLM_MAX_CALLS_PER_VIDEO`, default
  1; negative means unlimited).

Candidate selection is deterministic:

1. Aggregate collected OCR texts through `EnrichedMetadataExtractor`'s text
   aggregation helper.
2. Then consider individual frames ranked by OCR confidence and text length.
3. Remove normalized duplicates and candidates shorter than
   `LLM_MIN_TEXT_LENGTH` (32 by default).
4. Keep at most two candidates.

The extractor attempts at most two candidates, further capped by the remaining
budget. With defaults this is at most one logical metadata extraction per video.

LLM output is accepted only if it has at least one signal field and passes source
grounding checks (`frame_cleaner.py:801-856`):

- first/last names must be strings, at most 40 characters, contain no configured
  title/age/narrative tokens, and every token must occur in source OCR text;
- case number must occur in source OCR text after alphanumeric normalization.

The first valid candidate is merged. Invalid, empty, or failed results are
discarded; batch enrichment fails soft and returns no metadata.

### Final anonymization decision

The selected technique does not invoke another learned model
(`frame_cleaner.py:580-668`):

- `remove_frames`: remove the sampled indices classified as sensitive using
  FFmpeg. This classification only covers sampled frames.
- `mask_overlay`: mask the supplied valid overlay ROI, or a default fixed mask.
  It does not use the detected per-frame PHI boxes.
- `extract_only`: do not modify the video.
- unknown value: warn and return the output path.

## `ReportReader` report pipeline

### High-level flow

```text
construct ReportReader
  -> load spaCy extractors and probe provider model availability
  -> obtain text from caller, PDF text layer, or image Tesseract OCR
  -> if document text is shorter than 50 characters:
       render pages
       -> ensemble OCR or Tesseract
       -> optional Ollama vision OCR per page
       -> optional chunked LLM OCR correction
  -> local spaCy/regex metadata extraction
  -> only if local extraction has no signal: optional provider LLM extraction
  -> deterministic text anonymization
  -> optionally run EAST + OCR + spaCy/regex + custom PHI detection for PDF redaction
  -> emit SensitiveMeta-compatible metadata and provenance
```

### Initialization decisions

`ReportReader.__init__()` constructs four spaCy-backed extractors (patient,
examiner, endoscope, examination), `SensitiveRegionCropper`, and `Anonymizer`
(`report_reader.py:55-125`). The shared spaCy loading/fallback behavior is the
same as described for `FrameCleaner`.

Unlike `FrameCleaner`, report initialization always calls the LLM factory; it
does not gate provider discovery on `LLM_ENABLED` (`report_reader.py:127-155`).
If a current model is available, `llm_available` is true.

### Initial text acquisition

The input precedence is text, then PDF, then image (`report_reader.py:346-360`):

- Caller-provided text causes no OCR call.
- PDF text is extracted with pdfplumber, with no model inference
  (`report_reader_extraction.py:109-133`).
- A direct image input immediately makes a full-image Tesseract call with PSM 6
  and also obtains word boxes (`report_reader.py:362-369`;
  `ocr/ocr.py:235-259`).

Text-only requests skip the OCR fallback. For PDF/image requests, text with at
least 50 non-whitespace characters is accepted as-is. Shorter text triggers page
rendering and the fallback below (`report_reader.py:371-397`). Thus a direct image
can be OCRed a second time when its first Tesseract result is shorter than 50
characters.

If final OCR text is shorter than 10 characters, non-text-only processing returns
without metadata extraction or anonymization.

### Call RR-1: report fallback OCR

For every rendered page, `_ocr_image()` chooses conventional OCR as follows
(`report_reader.py:410-459`):

1. When `use_ensemble=True`, call `ensemble_ocr()`.
2. If ensemble output is empty or ensemble throws, call full-image Tesseract with
   PSM 6.
3. With `use_ensemble=False`, call Tesseract directly.

#### Ensemble model calls and selection

`ensemble_ocr()` attempts three OCR sources (`ocr/ocr_ensemble.py:89-222`):

- Tesseract LSTM (`--oem 1 --psm 6`), scored by mean Tesseract confidence;
- TrOCR `microsoft/trocr-base-str`, scored by `min(1, text_length / 500)`;
- Donut, scored from sentence and meaningful-word counts.

TrOCR loads through the singleton `ModelService`, prefers CUDA, falls back to CPU
on CUDA OOM, and itself falls back to Tesseract when dependencies/model loading
are unavailable (`ocr/ocr.py:262-312`; `model_service.py:221-293`). Generation is
deterministic beam search with five beams and up to 512 new tokens.

Each ensemble member fails independently to an empty result. The highest score is
selected. If its text is shorter than 20 characters, results are combined line by
line by choosing the longest corresponding line. English spell correction is
then applied to nonnumeric lines when the spellchecker is installed.

Current implementation note: the Donut import in the ensemble points to
`lx_anonymizer.ocr_donut`, while the repository module is
`lx_anonymizer/ocr/ocr_donut.py`. Unless another compatibility module is present
at runtime, this attempt is caught as a failed ensemble member.

#### Optional RR-1b: Ollama vision OCR

After conventional OCR, one vision call per page is attempted only when
`LLM_ENABLED`, `OLLAMA_OCR_ENABLED`, provider `ollama`, and
`ollama_available` are all true (`report_reader.py:437-459`). It uses the same
literal-transcription prompt and candidate-guided Ollama call as frame vision
OCR. Non-empty vision text replaces conventional text; failure retains it.

### Call RR-2: optional chunked LLM OCR correction

After all fallback pages are joined, OCR correction is considered
(`report_reader.py:461-498`; `llm/llm_service.py:60-136`):

- empty text: skip;
- fewer than `REPORT_OCR_CORRECTION_MIN_TEXT_LENGTH` characters (120 default):
  skip;
- provider unavailable: skip;
- otherwise split into 2,048-character chunks and make one chat inference per
  chunk.

The system prompt asks for correction of German medical report OCR while
preserving names, dates, and identifiers. Temperature is zero. A failed chunk is
replaced with its original text. The complete corrected result is accepted only
when it differs and is longer than half the original; otherwise all original OCR
text is retained.

This correction gate checks `llm_available`, not `LLM_ENABLED` or the request's
`use_llm`. Consequently `use_llm=False` disables report metadata LLM extraction
but does not disable OCR correction, and global `LLM_ENABLED=False` does not by
itself disable correction if model discovery succeeded.

### Call RR-3: local spaCy/regex metadata extraction

Local extraction always runs before an optional metadata LLM
(`report_reader.py:503-552`; `report_reader_extraction.py:135-303`):

1. Run `PatientDataExtractor` over the full text.
2. Accept it only if first or last name has signal.
3. Otherwise run it line-by-line only on lines matching the patient-line regex.
4. Otherwise use the dedicated regex fallback.
5. On matching lines, run spaCy/rule extractors for examiner and endoscope data
   and regex-first extraction for examination data.

Dates are normalized, results are validated through typed report models, and the
PDF SHA-256 is added. This path is based primarily on spaCy tokenization plus
explicit matchers and regexes, even though provenance labels it `spacy` and
`regex`.

### Call RR-4: optional report metadata LLM

Request-level metadata LLM use is `request.use_llm` when explicitly supplied;
otherwise it inherits `llm_available` (`report_reader.py:500-501`). The actual
inference is attempted only when:

- report text has at least 10 characters;
- provider LLM use is selected;
- **local extraction has no signal in any tracked field**; and
- text has at least `REPORT_LLM_MIN_TEXT_LENGTH` characters (64 default).

Tracked signal fields include patient identifiers, gender, examination
date/time, examiner names, and center (`report_reader.py:654-680`). Any one local
signal skips the LLM entirely, even if other fields are missing.

When attempted, `extract_report_meta_with_llm()` makes one logical call through
the shared structured metadata path (`report_reader_extraction.py:304-347`). A
valid result is merged through `SensitiveMeta`. Empty or failed LLM output falls
back to the already-used local extraction path. There is no report-specific
source-grounding validation equivalent to `FrameCleaner`'s name/case checks.

### Deterministic text anonymization

`anonymize_report()` does not call a learned model. It passes extracted metadata,
cutoff markers, locale, and replacement-name pools to the text anonymizer
(`report_reader_extraction.py:349-362`).

### Calls RR-5 through RR-8: optional PDF visual redaction

These calls occur only when `create_anonymized_pdf=True` and a PDF path is present
(`report_reader.py:554-587`; `anonymization/anonymizer.py:367-415`). Each rendered
page follows this additive detector path:

1. **Custom PHI ONNX detector (RR-5):** the same optional detector used by
   `FrameCleaner`; disabled by default.
2. **EAST text detector (RR-6):** one OpenCV DNN forward pass over a 640×640 image
   with minimum confidence 0.5 in this call path
   (`anonymization/anonymizer.py:208-255`). EAST's verified frozen graph is loaded
   or downloaded as needed (`text_detection/east_text_detection.py:57-65,182-242`).
3. **Tesseract on EAST boxes (RR-7):** use tesserocr when available, otherwise
   pytesseract, to associate text with localized boxes
   (`anonymization/anonymizer.py:61-74,233-243`).
4. **spaCy/rule + regex classification (RR-8):** `SensitiveRegionCropper`
   extracts patient data and applies patterns for names, DOB, case number, social
   security-like values, phone, address, and examiner names. Matching word boxes
   are expanded by a 20-pixel margin and must be at least 100×30 pixels
   (`anonymization/sensitive_region_cropper.py:57-97,212-299`).

EAST-derived sensitive regions and custom detector regions are merged, converted
to PDF coordinates, and permanently redacted with black annotations. If the
EAST-based path raises, full-image Tesseract word boxes plus the same sensitive
classifier are used as fallback (`anonymization/anonymizer.py:257-292`). A custom
PHI detector configured as required is not swallowed by that fallback.

The extracted `report_meta` is currently not used to decide redaction regions;
redaction re-detects content from page images.

### Alternate report cropping and visualization paths

`ReportReader.process_report_with_cropping()` first runs the normal
`process_report()` flow and then, when cropping is enabled and a PDF is present,
uses `SensitiveRegionCropper.crop_sensitive_regions()`
(`report_reader_extraction.py:389-543`). This is a separate path from the EAST
redactor:

1. Render every selected PDF page.
2. Run full-image Tesseract once per page to obtain word boxes.
3. Run the same spaCy/rule and regex sensitive-region classifier described under
   RR-8.
4. Save each accepted region as a crop.

If crops were found, `create_anonymized_pdf_with_crops()` renders the PDF again
and repeats full-image Tesseract plus sensitive-region classification once per
page before drawing black rectangles
(`anonymization/sensitive_region_cropper.py:440-517,552-630`). It does not reuse
the first pass's region coordinates, and it does not invoke EAST or the custom
PHI detector.

`ReportReader.create_visualization_report()` is a diagnostic public helper rather
than part of `process_report()`. It performs full-image Tesseract and the same
spaCy/rule + regex classification for the first page by default, or every page
when `visualize_all_pages=True` (`report_reader_extraction.py:545-577`).

## Model-related code not on the current entrypoint paths

The following components should not be counted as calls made by the two primary
workflows:

| Component | Current status |
| --- | --- |
| `BestFrameText` / `EnhancedBestFrameText` | Neither is instantiated or called by `FrameCleaner.clean_video()`. Best-frame selection there is confidence, then text length. |
| CRAFT | Implemented in `text_detection/craft_text_detection.py` and imported by general OCR utilities, but not called from either primary entrypoint path described above. |
| MiniCPM | `FrameCleaner` stores its constructor flags only; no MiniCPM load or inference follows. |
| EAST in frame analysis | Not called by `FrameCleaner`; the `east_ocr` frame observation tag is added whenever OCR text exists, regardless of the actual RapidOCR/tesserocr/pytesseract backend (`frame_cleaner.py:936-953`). |
| `FrameCleaner.extract_metadata()` | Public helper that prefers LLM then spaCy, but `clean_video()` does not call it. Video-level LLM use goes through `_extract_enriched_metadata_batch()`. |
| `FrameCleaner._should_attempt_llm()` | Defines per-text deduplication, minimum length, budget, and completeness checks, but has no caller in the current repository. |
| Smart-sampling LLM methods in `llm_extractor.py` | Available utilities, but neither primary entrypoint calls them. |
| Local Phi-3/Phi-4 loader in `ModelService` | The loader exists, but these entrypoints use provider REST APIs for metadata/correction and do not call it. |

## Call-count summary

Counts below are logical inference attempts before internal HTTP retries or model
fallbacks.

| Workflow stage | Normal maximum under defaults |
| --- | ---: |
| Frame conventional OCR | one cascade result per sampled frame, up to 24 in balanced mode |
| Frame Ollama vision OCR | up to one per high-quality sampled frame, therefore up to 24 by default |
| Frame custom PHI detector | zero by default; one per sampled frame when configured |
| Frame video-level metadata LLM | at most one per video by default |
| Report direct image initial OCR | one Tesseract call |
| Report fallback conventional OCR | one selected path per rendered page; ensemble can attempt Tesseract + TrOCR + Donut |
| Report Ollama vision OCR | up to one per fallback-rendered page |
| Report OCR correction LLM | one per 2,048-character chunk after pages are joined |
| Report metadata LLM | zero when any local signal exists; otherwise one logical extraction |
| Optional report redaction | per page: optional PHI detector + EAST + OCR on each EAST box; full-image Tesseract fallback on failure |
| Alternate report cropping | one full-image Tesseract + classifier pass per page to create crops; when crops exist, a second pass per page to draw the PDF |
| Visualization helper | one full-image Tesseract + classifier pass on the first page by default, or each requested page |

Provider metadata calls can exceed these logical counts because
`LLMMetadataExtractor` retries a failed HTTP request twice and can then switch to
another available model. Report OCR correction and vision calls do not have that
retry wrapper; their callers retain original/conventional text on failure.
