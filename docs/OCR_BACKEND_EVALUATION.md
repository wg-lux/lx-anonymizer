# OCR Backend Matrix Evaluation

This guide describes how to run the split-modality OCR benchmark and generate
publication-ready figures from the result JSONL.

The evaluator ranks OCR pipelines by empirical anonymization utility, not by
backend-specific confidence scores. It evaluates video-frame pipelines and
report-document pipelines separately.

## 1. Prepare the Environment

From a source checkout:

```bash
uv sync --extra dev
```

For live OCR runs, install the OCR/runtime extras that match the machine:

```bash
uv sync --extra cpu
# or, on CUDA hosts:
uv sync --extra gpu
```

For figure generation:

```bash
uv pip install matplotlib
```

## 2. Build the Golden-Set Manifest

The command expects a JSONL or JSON-array manifest. Every item must point to an
absolute source path and must include human-adjudicated ground truth for the
five standardized PHI fields.

Minimal JSONL row:

```json
{"sample_id":"report-001","input_type":"report","source_path":"/abs/path/report.pdf","ground_truth":{"first_name":"Alice","last_name":"Doe","dob":"1980-01-02","casenumber":"E 123","examination_date":"2024-03-04","bounding_boxes":[{"page_index":0,"x1":72,"y1":88,"x2":320,"y2":132}],"text":"Patient Alice Doe, DOB 1980-01-02, Case E 123"}}
```

Supported `input_type` aliases:

- Video frames: `video`, `frame`, `video_frame`, `image_frame`
- Reports: `report`, `text_report`, `text_report_document`, `pdf`, `document`

For video V1 fixed-ROI evaluation, include `roi` or `fixed_roi`:

```json
{"sample_id":"frame-001","input_type":"video_frame","source_path":"/abs/path/frame.png","roi":{"patient":{"x":40,"y":24,"width":420,"height":80}},"ground_truth":{"first_name":"Alice","last_name":"Doe","dob":"1980-01-02","casenumber":"E 123","examination_date":"2024-03-04","bounding_boxes":[{"x":40,"y":24,"width":420,"height":80}],"text":"Alice Doe 02.01.1980 E 123 04.03.2024"}}
```

Use 30 to 50 curated rows for the publication run, balanced across:

- electronic-text PDFs
- scanned and distorted historical reports
- high-contrast overlay frames
- low-contrast frames
- motion-blurred frames
- ultra-small overlay fonts
- clean frames with no overlay PHI
- report screenshots

Clean frames should set all five PHI fields to `null`, `bounding_boxes` to an
empty list, and `text` to an empty string or non-PHI reference text.

## 3. Run the Evaluation

Write machine-readable rows to a file:

```bash
python manage.py evaluate_ocr_backend_matrix \
  --golden-set=/abs/path/golden-set.jsonl \
  --output-jsonl=results/ocr-backend-matrix.jsonl
```

Run a subset while debugging:

```bash
python manage.py evaluate_ocr_backend_matrix \
  --golden-set=/abs/path/golden-set.jsonl \
  --pipelines=V1,V2,R1,R2 \
  --output-jsonl=results/debug.jsonl
```

By default the JSONL avoids writing raw recognized text and predicted PHI
values. For local, access-controlled error analysis only:

```bash
python manage.py evaluate_ocr_backend_matrix \
  --golden-set=/abs/path/golden-set.jsonl \
  --output-jsonl=results/with-sensitive-output.jsonl \
  --include-sensitive-output
```

Do not use `--include-sensitive-output` for shared artifacts, paper
supplements, issue comments, or logs.

## 4. Interpret the JSONL

The output contains two row types:

- `sample_result`: one row per sample and pipeline execution
- `summary`: aggregated ranking rows grouped by modality and pipeline

The primary ranking metric is `mean_utility_score` in summary rows.

The utility score is:

```text
U = 0.35 * R_PHI
  + 0.25 * A_field
  + 0.15 * S_text
  + 0.10 * C_box
  + 0.10 * S_speed
  + 0.05 * M_stab
```

Hard gate: if PHI field recall is below `0.95`, the sample utility score is
forced to `0.00`. This models a catastrophic privacy failure: speed or OCR
fluency cannot compensate for missing identifiers.

## 5. Generate Publication Figures

Create vector and raster figures from the evaluator JSONL:

```bash
python scripts/plot_ocr_backend_matrix.py \
  --input-jsonl=results/ocr-backend-matrix.jsonl \
  --output-dir=results/figures \
  --formats=pdf,svg,png
```

Generated figures:

- `ocr_backend_utility_ranking.*`: ranked utility by pipeline and modality
- `ocr_backend_component_heatmap.*`: component-level score heatmap
- `ocr_backend_latency_tradeoff.*`: utility-vs-latency scatter with pipeline means

For manuscripts, prefer the PDF or SVG outputs. PNG files are useful for slide
decks and quick previews.

## 6. Figure Styling Recommendations

Use a consistent figure numbering scheme:

- Figure 1: Utility ranking by modality
- Figure 2: Component score heatmap
- Figure 3: Utility and throughput trade-off

Recommended caption language:

> OCR pipelines were ranked by anonymization utility, a weighted score combining
> PHI recall, normalized field accuracy, text similarity, bounding-box coverage,
> throughput, and process stability. A hard gate set utility to zero for samples
> with PHI recall below 0.95.

Report both mean utility and PHI recall. In privacy-sensitive evaluation, a fast
pipeline with low recall should be treated as unsafe even if its throughput is
excellent.

## 7. Reproducibility Checklist

Record these values for a paper or supplement:

- Git commit SHA
- manifest SHA-256
- `python --version`
- `tesseract --version`
- installed OCR backends and model versions
- CPU/GPU model and memory
- whether `--include-sensitive-output` was disabled for shared outputs
- exact evaluation and plotting commands

Example manifest checksum:

```bash
sha256sum /abs/path/golden-set.jsonl
```

Example command provenance:

```bash
python manage.py evaluate_ocr_backend_matrix --help
python scripts/plot_ocr_backend_matrix.py --help
```
