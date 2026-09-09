# LX Anonymizer documentation

`README.md` is the canonical entry point. It contains the supported installation
paths, the quickstart, the public CLI and Python APIs, and the safety boundary
between this package and its caller. This directory contains the longer-lived
guidance that is useful after the first successful run.

## Choose a guide

### Users and operators

- [Configuration reference](../SETTINGS.md) — environment variables, model
  providers, OCR, detector, and video settings.
- [Development environment and Nix outputs](../Devenv.md) — use the repository as
  a `devenv` or Flake input and understand the package variants.
- [Release guide](../RELEASING.md) — maintainer checklist for validation,
  packaging, TestPyPI, and production publication.

### Contributors

- [Contributing](../CONTRIBUTING.md) — local setup, checks, pull requests, and
  project communication.
- [Model calls and decision flow](MODEL_CALLS_AND_DECISIONS.md) — the OCR,
  metadata, detector, and anonymization decisions made by the active pipelines.
- [Video import concurrency contract](VIDEO_IMPORT_CONCURRENCY_CONTRACT.md) —
  the boundary between `lx-anonymizer` and `endoreg-db` for video attempts,
  temporary resources, cancellation, and cleanup.
- [Report pipeline concurrency contract](report_pipeline_concurrency_contract.md)
  — the corresponding report-processing contract.
- [NVENC acceleration](NVENC_ACCELERATION.md) — hardware encoding behavior and
  CPU fallback.

### Evaluation and model development

These documents record reproducible engineering work; they are not product
quality claims or substitutes for clinical validation.

- [OCR backend evaluation](OCR_BACKEND_EVALUATION.md) — how to run and interpret
  the backend matrix.
- [MIDI-B evaluation record](midi-b-evaluation-2026-07-14.md) — dated findings,
  limitations, and acceptance-gate recommendations.
- [PHI-region detector training](phi-region-detector-training.md) — dataset,
  synthetic-data, and training workflows.
- [PHI-region detector model README](../models/phi-region-detector/README.md) —
  the model artifact and inference contract.

## Documentation conventions

- Keep user-facing behavior, supported commands, and public API examples in
  `README.md`.
- Keep durable operational contracts and developer procedures in this directory
  or in the linked repository-owned guide.
- Keep dated measurements and experiment results clearly marked as records;
  update them with a new dated record when the evidence changes.
- Do not document secrets, patient data, generated artifacts, or local absolute
  paths as required setup.

When code changes alter a command, public API, configuration variable, output
contract, or safety invariant, update the relevant guide in the same change.
