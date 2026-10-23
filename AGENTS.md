# This module has a strong focus on anonymizing video and image data

- Main Entrypoint Video: /lx_anonymizer/frame_cleaner/__init__.py
- Main Entrypoint for Reports: /lx_anonymizer/report_reader/__init__.py
- Study Data about the LLM usage in the pipeline: ./study_data

## The pipeline contains

Multi-Detector Cascade: EAST, Tesseract, and CRAFT for robust localization.

Fuzzy-Spatial Coupling: text correction and bounding box resizing.

Heuristic Selection: BestFrameText scoring and LLM integration for identifying the most relevant data.

Anonymization Strategy: Blurring + NER.

Standardized Output: /home/admin/lx-anonymizer/lx_anonymizer/sensitive_meta_interface.py

## Source of Truth & Context Protocol

- Start with the root `structure.yml`, then follow `submodules` into the domain
  relevant to the task. Read its entrypoints and contracts before inspecting
  implementation. Use scoped searches in that tree rather than broad repository
  grepping; follow dependency manifests when the task crosses a boundary.
- Manifests are navigation and boundary metadata. Code, shared `lx_dtypes`
  contracts, and the referenced canonical specifications remain authoritative
  for behavior. Preserve existing feature assessments and custom manifest fields.
- Whenever adding, removing or renaming files or changing architectural
  boundaries, update the corresponding `structure.yml` files in the same
  changeset. Record important files in entrypoints or the optional `files`
  inventory; small helpers remain covered by their nearest manifest. Keep child
  links and parent links consistent. Do not generate a manifest for every folder.
- Scope delegated work to a manifest-owned directory tree, its contracts and a
  focused verification command. Coordinate before editing another domain or
  shared interfaces; delegation itself requires task authorization.
- Before concluding any prompt or submitting a pull request, run
  `python scripts/lint_structure.py` (or `uv run python scripts/lint_structure.py`)
  and resolve all reported drift. For code changes, strict Pyright precedes pytest
  and `uv sync --extra dev` is required before pytest.
- See [agent context guidance](docs/agent_context.md) for schema and path rules.

## Mandatory video concurrency contract

Before changing `frame_cleaner/`,
`video_processing/`, native Rust video helpers, FFmpeg or FFprobe execution,
temporary video files, output naming, cancellation, retry, or resource
parallelism, read and preserve:

- `docs/VIDEO_IMPORT_CONCURRENCY_CONTRACT.md`;
- the canonical sibling-repository contract
  `/home/admin/endoreg-db/docs/video_import_concurrency_contract.md`;
- `/home/admin/endoreg-db/docs/video_storage_normalization.md`;
- `/home/admin/endoreg-db/feature-tracking/VideoStorageNormalization.yml`.

`endoreg-db` owns durable attempts, leases, fencing, encrypted storage,
publication, and cleanup. `lx-anonymizer` processes exactly one immutable source
into one unpublished attempt-owned candidate. Do not add competing persistence,
locking, publication, or retry authority here.

A `FrameCleaner` instance contains mutable per-video state and must not be used
concurrently. Create one instance per attempt until run state is separated into
a typed invocation object and shared model components have explicit thread- and
process-safety tests.

Every FFmpeg process, named pipe, temporary directory, and output path must have
one attempt owner, bounded cancellation, and deterministic cleanup. Integrity
failures must raise loudly; boolean failure returns and silent fallback must not
turn a partial artifact into apparent success.

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
