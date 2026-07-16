# PHI region detector training

The custom detector is a recall-oriented, additive safeguard. Its predictions
are combined with the existing EAST/OCR path; they do not replace it.

## MIDI-B data

MIDI-B is distributed as DICOM validation and test collections with SQLite
answer keys. It is not a ready-made YOLO training directory. The pixel actions
in an answer key contain `top_left` and `bottom_right` coordinates and can be
converted into YOLO labels, but the official test collection must remain held
out if results are to be reported as MIDI-B benchmark results.

For model development, use a separate patient-level training split (for
example, the public Pseudo-PHI-DICOM data or locally generated synthetic PHI),
reserve MIDI-B validation for threshold selection, and use MIDI-B test once for
the final evaluation. Never split individual frames from the same patient or
series between train and validation.

The files currently downloaded under `midi-b-dataset/` are TCIA manifests, not
the DICOM pixels or answer-key databases. Downloading the corresponding
synthetic collections and answer keys is required before label conversion.

## MIDI-B pixel evaluation

Install the evaluation dependency and score a trained ONNX detector against the
validation answer-key database:

```bash
uv sync --extra dev --extra evaluation
lx-anonymizer-evaluate-midi-b \
  --dataset-root /home/admin/TCIA-MIDI-B-Synthetic-Validation_20250502/MIDI-B-Synthetic-Validation \
  --answer-db /absolute/path/to/midi-b-validation-answer-key.db \
  --detector phi-model \
  --model-path /absolute/path/to/best.onnx \
  --output study-data/midi-b/phi-model-evaluation.json
```

For a quick integration check or OCR baseline, use `--detector tesseract` and
optionally `--max-instances 10`. The JSON report contains overall and
per-modality precision, recall, F1, mean best IoU, and union coverage of each
ground-truth PHI box. Multi-frame DICOM instances currently evaluate their
first frame.

This command evaluates burned-in pixel-PHI localization only. It does not score
DICOM element actions, private-tag handling, UID/date transformations, output
conformance, or the official MIDI-B end-to-end anonymization contract.

## YOLO contract

Use one class named `phi` unless there is a concrete operational need for
separate PHI types. A minimal dataset file is:

```yaml
path: /absolute/path/to/phi-yolo
train: images/train
val: images/val
test: images/test
names:
  0: phi
```

Each image must have a matching label file. Keep empty label files for negative
images; excluding them teaches the detector that every image contains PHI and
causes excessive false positives.

## Generate OpenCV synthetic frames

The repository includes a deterministic generator that renders synthetic PHI
with OpenCV onto DICOM or raster medical-image backgrounds. It writes exact
YOLO boxes from `cv2.getTextSize`, empty labels for negative frames, a JSONL
provenance manifest, a summary, and `dataset.yaml`. Source files are read-only.

Install the DICOM-capable generator with `uv sync --extra dev --extra
evaluation`. Add `--extra cpu` or `--extra gpu` when installing the training
stack itself.

The MIDI-B starter dataset in this workspace was generated with:

```bash
lx-anonymizer-generate-phi-data \
  --source-root /home/admin/TCIA-MIDI-B-Synthetic-Validation_20250502/MIDI-B-Synthetic-Validation \
  --output-root /home/admin/lx-anonymizer/study_data/phi_synthetic_midi_b_seed0 \
  --names-source /home/admin/lx-anonymizer/study_data/gold.json \
  --seed 0 \
  --frames-per-patient 1 \
  --negative-fraction 0.15 \
  --max-dimension 1024
```

Use `--max-patients` for a smaller smoke dataset or increase
`--frames-per-patient` to expand the corpus. The generator automatically avoids
multi-frame images and sources over
4,194,304 decoded pixels so compressed DICOMs cannot unexpectedly exhaust
memory during routine generation. The split is 60% train, 20% validation, and
20% test by default. A
patient is assigned to exactly one split, so frames from one patient cannot
leak across partitions. The output directory must be absent or empty; the
generator will not overwrite an existing dataset.

This synthetic dataset trains overlay localization, not semantic recognition
of every real burned-in PHI style. Keep the untouched MIDI-B answer-key data as
the external evaluation set.

Train deterministically with:

```bash
lx-anonymizer-train-phi \
  --dataset-yaml /home/admin/lx-anonymizer/study_data/phi_synthetic_midi_b_seed0/dataset.yaml \
  --output-dir /absolute/path/to/runs \
  --base-model yolov8n.pt \
  --epochs 100 \
  --input-size 640 \
  --seed 0
```

The command exports ONNX plus a metadata JSON containing the checksum and exact
runtime settings. Runtime preprocessing defaults to `letterbox`, matching
Ultralytics training while preserving non-square medical-image geometry.

For privacy use cases, select the confidence threshold from the validation
precision-recall curve based on the required PHI recall, and report false
negatives per modality and manufacturer. Do not select a model from aggregate
mAP alone.
