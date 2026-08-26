# Video Import Concurrency Contract for `lx-anonymizer`

This document defines how `lx-anonymizer` participates safely in concurrent
video imports. The canonical cross-repository contract and all implementation
status are owned by the `endoreg-db` feature
`video_storage_normalization`, specifically:

- [cross-repository concurrency contract](https://github.com/wg-lux/endoreg-db/blob/main/docs/video_import_concurrency_contract.md);
- [video storage normalization runbook](https://github.com/wg-lux/endoreg-db/blob/main/docs/video_storage_normalization.md);
- [VideoStorageNormalization feature definition](https://github.com/wg-lux/endoreg-db/blob/main/feature-tracking/VideoStorageNormalization.yml).

This document is an implementation reference, not an independent roadmap or
completion tracker. It may make the component rules stricter, but it may not
reassign durable state, storage, publication, retry, or cleanup ownership. If
the documents conflict, the `endoreg-db` contract and feature definition
prevail. Boundary changes must update both contracts and record the paired
repository revisions in review evidence.

## Component Role

`lx-anonymizer` is an attempt-scoped compute library. It receives one immutable
video source, applies clinical masking and metadata extraction, and returns one
unpublished candidate with typed metadata.

It does not own:

- durable import state;
- database transactions;
- content-hash deduplication;
- leases, heartbeats, or fencing tokens;
- encrypted storage routing;
- canonical publication;
- retry, quarantine, or cleanup policy.

Those responsibilities remain in `endoreg-db`. A successful
`FrameCleaner.clean_video` return never means that a canonical artifact may be
published.

## Current Concurrency Limitation

`FrameCleaner` contains mutable per-video state:

- `frame_collection`;
- `frame_observations`;
- `ocr_text_collection`;
- language-model call counts and seen text;
- `SensitiveMeta`;
- `current_video_total_frames`.

Calling one instance concurrently for multiple videos is prohibited. Until
this state is moved into a typed invocation object, callers must construct one
`FrameCleaner` per attempt. Tests must treat accidental instance sharing as a
contract violation rather than relying on `_reset_run_state`.

Model initialization may eventually be separated from run state. A shared
model cache is allowed only when its thread and process safety is explicit,
tested, and independent of mutable patient or video data.

## Required Invocation Shape

A future typed invocation model should contain:

- opaque attempt identifier;
- immutable input path or already-open descriptor;
- caller-owned attempt output path;
- anonymization technique and versioned quality profile;
- normalized endoscope and sensitive-region coordinates;
- cancellation probe;
- bounded phase timeouts;
- optional resource allocation identity.

The result should contain:

- candidate path;
- typed sensitive metadata;
- anonymizer provenance;
- frame observations and protected-health-information region proposals;
- frames inspected and total frames observed;
- phase timings;
- encoder and hardware selection;
- explicit warnings;
- output byte count.

Invalid or incomplete input must raise a typed error. Returning a boolean or
empty dictionary must not hide a failed integrity boundary.

## Input and Output Rules

1. Treat the input as immutable and read-only. Never rename, truncate, append
   to, delete, or rewrite it.
2. Reject missing files, symbolic links, non-regular files, and empty files at
   the invocation boundary.
3. Use only the caller-provided output path for `endoreg-db` integration.
4. The output path must be unique to the attempt and retain its media suffix so
   FFmpeg can select the container.
5. Do not derive a shared output name from the source when an explicit output
   was supplied.
6. Refuse a pre-existing output unless the caller provides typed proof that it
   belongs to the same idempotent attempt.
7. FFmpeg's overwrite option may target only that proven attempt-owned path.
8. Write no plaintext media outside the caller-approved encrypted staging
   boundary.
9. Return an unpublished candidate. Do not move it into a canonical location.
10. On failure or cancellation, remove only invocation-owned partial files and
    never touch the input or another attempt.

## FFmpeg Process Ownership

Every FFmpeg or FFprobe process belongs to exactly one invocation.

- Build argument arrays without a shell.
- Retain the process handle and start the process in a controllable process
  group.
- Apply a phase-specific timeout.
- Capture bounded standard error suitable for diagnosis without exposing
  patient data.
- Poll the cancellation probe during long operations.
- On cancellation, lease loss, or timeout, terminate the process group, wait a
  bounded grace period, force termination if necessary, and reap every child.
- Give named pipes and temporary directories unique attempt-derived names.
- Close pipe descriptors on every success and failure path.
- Validate process exit status and require a non-empty candidate.
- Do not turn encoding, probing, or masking errors into `False` followed by a
  different integrity path.

Hardware fallback may change the encoder only when the selected clinical
profile explicitly permits it. It must be named, logged, tested, and produce a
candidate that still undergoes `endoreg-db` validation.

## Native Rust Boundary

The native module is suitable for bounded pure transformations and
data-parallel analysis. It must not own workflow state.

- Long native work releases the Python Global Interpreter Lock (GIL).
- Python stubs describe every native input and output.
- The module exposes a compatibility version and named capabilities.
- Integrity-relevant native errors propagate to the caller.
- Production may require a compatible native capability set and fail startup
  when it is missing.
- A Python fallback is development-only unless explicitly approved, observable,
  and behaviorally equivalent.
- Rayon thread counts must be configurable so native parallelism does not
  multiply uncontrollably across worker processes.
- Native code must not access Django, publish files, route storage, acquire
  durable locks, or delete artifacts.

## Resource Isolation

Concurrency belongs between attempts. An individual invocation remains a
bounded sequential publication candidate.

- `endoreg-db` controls admission and worker concurrency.
- `lx-anonymizer` reports resource requirements and measurements; it does not
  start an unbounded internal task pool.
- Graphics processor selection must be explicit. Concurrent invocations may
  not all assume device zero without allocation.
- OpenCV, Optical Character Recognition, language-model, Rayon, and FFmpeg
  thread counts must have bounded configuration.
- Temporary byte use must remain within the caller's attempt directory.
- Memory-heavy frame collections must have typed maximum sizes.
- Backpressure and capacity failures raise explicitly; they do not reduce
  anonymization quality silently.

## Determinism and Retry

Given the same immutable source generation, profiles, model versions, and
random seed, retry must produce semantically equivalent masking and metadata
results.

- Reset all per-video state before work begins.
- Prefer a newly constructed run-state object so partial reset cannot leak
  state.
- Record model, detector, language-model, encoder, and configuration versions
  in provenance.
- Record any nondeterministic seed.
- Never reuse partial output as successful input.
- A retry writes to a new attempt path unless the caller proves idempotent
  ownership of the existing path.

## Required Tests

Changes to video execution must add or preserve tests for:

1. two separate `FrameCleaner` instances running in parallel without metadata
   or frame-observation leakage;
2. deliberate concurrent reuse of one instance being rejected;
3. unique outputs and temporary paths for simultaneous attempts;
4. cancellation and timeout terminating all FFmpeg children;
5. a failed invocation leaving no ambiguous non-empty candidate;
6. symbolic-link, missing, empty, and pre-existing-output rejection;
7. bounded Rayon, OpenCV, Optical Character Recognition, and FFmpeg concurrency;
8. central processing unit and graphics processing unit paths producing
   profile-valid candidates;
9. deterministic retry provenance;
10. missing or incompatible required native capability failing closed.

Use process-level tests for FFmpeg and worker behavior. Thread-only tests are
not sufficient evidence for production concurrency.

## Review Questions

Before merging a video-path change, reviewers must be able to answer:

- Which object owns mutable state for this invocation?
- Can two workers select the same output, pipe, or temporary directory?
- Who owns cancellation and every child process?
- Can an obsolete worker still write after losing its `endoreg-db` lease?
- Does any fallback weaken masking, timeline, or storage integrity?
- Is native parallelism bounded across worker processes?
- Are partial artifacts distinguishable from validated candidates?
- Does the result contain enough typed provenance for `endoreg-db` validation?

If any answer is unknown, the change is not ready for the production video
workflow.
