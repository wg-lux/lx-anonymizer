# This module has a strong focus on anonymizing video and image data
- Main Entrypoint Video: /lx_anonymizer/frame_cleaner.py
- Main Entrypoint for Reports:  /lx_anonymizer/report_reader.py
- Study Data about the LLM usage in the pipeline: ./study-data

## The pipeline contains

Multi-Detector Cascade: EAST, Tesseract, and CRAFT for robust localization.

Fuzzy-Spatial Coupling: text correction and bounding box resizing.

Heuristic Selection: BestFrameText scoring and LLM integration for identifying the most relevant data.

Anonymization Strategy: Blurring + NER.

Standardized Output: /home/admin/lx-anonymizer/lx_anonymizer/sensitive_meta_interface.py

## Type-first engineering rules

Use types as a primary safety rail. This project uses strict Pyright. For code
changes, run `.devenv/state/venv/bin/pyright` before pytest and treat type
failures as implementation failures, not cleanup. If a proposed diagnosis would
imply a type error, ask whether the types should have caught it and tighten the
type boundary where appropriate.

### Type expectations

- Wherever possible, typed files should live in `lx_dtypes` and use the existing
  `knowledge_base`.
- Prefer explicit function signatures, return types, typed dataclasses, enums,
  `TypedDict`, and Pydantic models over unstructured dictionaries.
- Annotate class attributes in the class body when they are assigned later.
- Avoid broadening types to make a failing test pass. Optional and union types
  need a concrete domain reason.
- Avoid `Any`. When interfacing with framework or external-library dynamic data,
  validate or narrow it at the boundary and pass typed objects inward.
- Use overloads or literal-discriminated helpers when inputs determine return
  types.

### Boundary and invariant rules

- Convert external input at the edge: request payloads, files, YAML/JSON,
  environment variables, command options, and third-party API responses should be
  normalized once and then represented with one typed internal shape.
- Define valid input invariants for non-trivial functions. Invalid input must
  raise loudly rather than being silently ignored.
- Prefer pure functions and returned values for transformation logic. Keep
  database writes, filesystem writes, network calls, and object mutation at
  explicit workflow boundaries.

### Exception handling

- Avoid broad `except Exception` outside request, command, job, or integration
  boundaries.
- Keep `try` blocks narrow, usually around one operation, and catch specific
  exception classes.
- Do not add silent fallbacks. If fallback behavior is explicitly required, make
  it named, logged, tested, and safe for clinical/security invariants.

### Testing expectations

- Add parametrized tests for meaningful valid input variation.
- Add invalid-input tests for invariants and boundary validation.
- Prefer focused unit tests for pure logic and integration tests only where
  contracts cross services, persistence, filesystem, or API boundaries.
- For code changes, run `.devenv/state/venv/bin/pyright` before pytest. Before
  pytest, run `uv sync --extra dev`.
- Do not use one-off scripts as a substitute for reusable tests when the behavior
  is important.
