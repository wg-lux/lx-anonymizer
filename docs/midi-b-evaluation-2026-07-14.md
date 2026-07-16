# MIDI-B anonymization evaluation — 2026-07-14

## Executive result

`lx-anonymizer` cannot yet be assigned an official end-to-end MIDI-B
anonymization-quality score because the official answer-key SQLite database is
not present locally and the repository does not emit de-identified DICOM files.

The pixel-PHI detector is now benchmarkable: a separate typed evaluator reads
the MIDI-B answer database, renders the source DICOM pixels without changing
their coordinate system, runs either the custom PHI model or a Tesseract
baseline, and reports precision, recall, F1, IoU, ground-truth coverage, and
per-modality results. The existing image/PDF/video entrypoints were not changed.

This remains a partial readiness result, not a numeric quality result. Pixel
detection can be measured when the answer database is supplied; DICOM metadata
de-identification and standards compliance remain outside this evaluator.

## Changes made after the initial readiness run

- Normalized standard `pytesseract.image_to_data()` output at the typed boundary
  by selecting the six supported fields and rejecting missing required fields.
- Made unavailable CRAFT detection an explicit logged optional path.
- Corrected fuzzy OCR candidate handling so the native matcher receives a list
  once per image instead of a string once per phrase.
- Added `lx-anonymizer-evaluate-midi-b`, a non-mutating MIDI-B pixel detector
  evaluator, plus an optional `evaluation` dependency for `pydicom`.
- Added strict typed tests for the answer-key parser, DICOM rendering boundary,
  detection metrics, Tesseract normalization, and missing CRAFT behavior.
- Re-ran the actual rendered-DICOM image pipeline successfully with EAST and
  Tesseract (`4` OCR results, no sensitive-name detections in that control).

The initial failures below are retained as provenance for why these changes
were needed.

## Evaluation scope

- Source: `/home/admin/TCIA-MIDI-B-Synthetic-Validation_20250502`
- Manifest directory: `/home/admin/lx-anonymizer/midi-b-dataset`
- Snapshot during an active download: 216 patient directories, 271 series
  directories containing DICOM files, 22,384 `.dcm` files, approximately 6.0
  GB.
- Published complete validation set: 216 subjects, 280 series, 23,921 images.
- Repository entrypoint tested:
  `lx_anonymizer.pipeline_manager.process_images_with_OCR_and_NER`.
- Pixel reconnaissance: one DX control plus a stratified sample containing one
  US, one MG, one CR, and one additional DX instance.

The snapshot was incomplete and changing during evaluation. Results therefore
describe pipeline readiness and sampled behavior, not final benchmark
performance.

## Findings

### 1. Native DICOM processing: fail

Passing a downloaded `.dcm` instance to the pipeline raised:

```text
RuntimeError: Error in process_images_with_OCR_and_NER: Invalid file type.
```

The accepted types are JPG, JPEG, PNG, TIFF, and PDF. There is no workflow that
rewrites DICOM attributes, private elements, UIDs, dates, structured reports,
file meta, or pixel data and then emits a conformant DICOM object.

This prevents evaluation of the central MIDI-B contract: executing the required
action for each DICOM element while preserving standards compliance and research
utility.

### 2. Rendered pixel masking: initial failure, now remediated

A DX DICOM instance was windowed, normalized, rendered to PNG, and passed to the
same masking pipeline. EAST initialized successfully, but Tesseract detection
failed at the typed external-data boundary. `pytesseract.image_to_data()`
returned six standard fields that `TesseractOCRData` forbids:

```text
level, page_num, block_num, par_num, line_num, word_num
```

The resulting Pydantic validation error was wrapped as `RuntimeError`. This
boundary has now been normalized and tested. A subsequent rendered-DICOM smoke
run completed and produced output.

This boundary should be normalized before validation or the typed model should
explicitly represent the supported Tesseract payload. The failure is consistent
with the project's type-first policy: the external shape and internal contract
currently disagree at runtime.

### 3. Optional/runtime components

- Ollama is installed in the devenv profile, but no Ollama server was reachable
  and no Gemma 4 OCR inference ran.
- The verified EAST model was obtained and loaded.
- CRAFT/`hezar` is absent. The pipeline now logs this and continues with EAST
  and Tesseract.
- The environment reported that no PyTorch, TensorFlow, or Flax backend was
  available, preventing transformer model execution.

### 4. Limited OCR observations: insufficient for a quality claim

The rendered DX control contained visible acquisition/orientation text. Plain
Tesseract recognized labels including `PORTABLE` and `SEMI-ERECT`; it did not
recognize the DICOM Patient Name or Patient ID. The four-modality stratified
sample similarly produced no exact Patient Name or Patient ID matches in OCR
output.

These five observations are not pixel-PHI ground-truth results. A missing exact
match may mean the identifier was not burned into that image, or that OCR missed
it. Without the MIDI-B pixel answer key, it must not be counted as a true
negative.

### 5. Official MIDI-B scoring inputs are absent

The local manifest directory contains Retriever binaries and TCIA manifests,
but not the validation answer-key SQLite database, UID mapping, patient mapping,
or a curated output produced by `lx-anonymizer`. TCIA publishes the validation
answer key separately (approximately 2.6 GB).

The official MIDI validation script therefore cannot calculate required-action
accuracy or DICOM compliance for this run. Even with the answer key, it needs a
de-identified DICOM output tree, which the current repository does not create.

## Quality assessment

| Dimension | Result | Evidence |
| --- | --- | --- |
| DICOM input compatibility | Fail | `.dcm` rejected as invalid file type |
| Header/private-tag anonymization | Not implemented | No DICOM read/rewrite output path |
| UID/date transformation | Not implemented | No MIDI-B mapping-aware DICOM workflow |
| Pixel detector evaluation | Ready when answer DB is supplied | Typed evaluator and tests added |
| Rendered image pipeline | Smoke test passes | EAST + Tesseract completed on sampled DX image |
| Gemma 4 OCR | Not exercised | Ollama server unavailable |
| DICOM conformance | Not measurable | No output DICOM files |
| Research-image utility | Not measurable | No anonymized image set |
| Official required-action score | Not computable | No output and no local answer key |

## Recommended acceptance gate

Do not use the current pipeline to release MIDI-B-like DICOM data. Before a
quality benchmark is meaningful, the repository needs:

1. A typed DICOM boundary that reads, transforms, validates, and writes complete
   DICOM objects, including private and nested elements.
2. Explicit MIDI-B/DICOM PS3.15 action rules for identifiers, dates, UIDs,
   free-text attributes, and de-identification method markers.
3. ~~A corrected Tesseract payload boundary and a named, tested fallback when
   optional CRAFT is unavailable.~~ Implemented for Tesseract and CRAFT.
4. Pixel-mask provenance that maps every redaction to the original DICOM frame
   and preserves pixel encoding and modality semantics.
5. A complete validation download, the official answer key and mappings, and an
   end-to-end output tree scored with the official MIDI validation script plus
   manual review of pixel cases.

Until those gates pass, the correct evaluation status is **not ready for MIDI-B
anonymization**, rather than a numeric anonymization-quality score.

## References

- [TCIA MIDI-B collection and downloads](https://www.cancerimagingarchive.net/collection/MIDI-B-Test-MIDI-B-Validation/)
- [Official MIDI validation script](https://github.com/CBIIT/MIDI_validation_script)
- [NCI benchmark overview](https://www.cancer.gov/about-nci/organization/cbiit/news-events/news/2025/validate-and-test-your-dicom-de-identification-algorithms-using-midi-benchmark)
