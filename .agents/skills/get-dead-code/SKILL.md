---
name: get-dead-code
description: Inventory, investigate, classify, and optionally remove dead code across the lx-anonymizer repository. Use for repository-wide dead-code audits, unused files or symbols, unused imports or parameters, stale re-exports and compatibility wrappers, unreachable branches, Vulture findings, dead-code baseline failures, or requests to safely remove confirmed dead code.
---

# Get Dead Code

Produce an evidence-backed repository-wide dead-code and duplication inventory. Treat static-tool output as candidates, never as deletion authority. Default to read-only analysis; modify or remove code only when the user explicitly requests it.

When the request is to reduce code without changing behavior, also inspect for
duplicate type aliases, private functions, classes, and documentation. Use
Radon complexity measurements as a prioritization signal, not as deletion
authority. Before removing or consolidating a candidate, derive its callers
with repository search and, where runtime reachability matters, run an
isolated `python -m trace --count -C <absolute-output-directory>` invocation
against the narrowest reproducible command or test. Preserve the command,
working directory, revision, and relevant output as evidence; tracing one
scenario does not prove global non-use.

Behavior-preserving refactors require focused Arrange–Act–Assert tests for
each changed contract, including meaningful valid and invalid cases. Run
Pyright before the focused tests for any non-trivial change, then run the
repository dead-code and relevant quality guards. Do not claim unchanged test
coverage from a passing test alone: compare the affected test surface and
coverage evidence when coverage is part of the request.

## Establish scope and policy

1. Read `AGENTS.md` and any narrower applicable instructions.
2. Inspect `git status --short` and preserve unrelated user changes.
3. Read `feature-tracking/policy.yml`, `feature-tracking/CodeQuality.yml`, and `docs/code_quality.md` before changing tracked code-quality behavior or removing code.
4. Inventory source roots and language manifests with `rg --files`. Do not infer repository coverage from filenames alone.
5. State the exact files to change and what will remain unchanged before editing.

For a code-reduction cohort, record the baseline line count, Radon complexity
for affected functions or classes, and the candidate's static, dynamic, public,
and documentation consumers. Keep duplicate documentation only when it is a
deliberate audience-specific contract; otherwise nominate one owning source
and link or remove the duplicate as part of the same reviewed cohort.

## Run the existing Python guard

Use the repository's reviewed configuration as the primary Python scan:

```bash
devenv tasks run quality:dead-code
```

Read `quality/dead_code_baseline.yml` and `scripts/check_dead_code.py` to interpret new, stale, expired, and accepted findings. Do not update the baseline merely to make the guard pass.

If raw finding details are needed, run Vulture through the project virtual environment using the paths, exclusions, and confidence threshold from `quality/dead_code_baseline.yml`. Preserve the exact path, line, message, confidence, and size as evidence.

For non-Python source roots, use only analyzers already configured in their language manifests or repository tasks. Report an uncovered language as a coverage gap; do not install a new analyzer or claim repository-wide coverage without evidence.

## Investigate every candidate

For each file or symbol, gather multiple independent forms of evidence where applicable:

1. Locate its definition, direct references, imports, re-exports, aliases, tests, documentation, configuration, and historical compatibility notes with `rg` and `git`.
2. Inspect call sites and surrounding ownership boundaries rather than judging from the name or tool message.
3. Check dynamic consumers: Django application configuration, model and admin discovery, URL routing, signals, management commands, task registration, serializers, templates, string imports, plugin registries, reflection, dependency injection, and command entry points.
4. Check package contracts: `__init__.py`, `__all__`, type stubs, protocol signatures, public APIs, downstream repositories when available, and compatibility wrappers.
5. Treat migrations, settings, generated code, typing-only imports, framework callbacks, and test fixtures as special surfaces requiring explicit proof.
6. Distinguish an unused implementation from an intentionally retained parameter or import. Do not broaden types, add ignores, suppress errors, or create silent fallbacks.
7. Run the narrowest relevant import, collection, framework, or focused test check when static evidence is insufficient.
8. For duplicate aliases, private functions, classes, or documentation, compare
   signatures, annotations, defaults, exceptions, side effects, ordering,
   framework registration, and public import paths before consolidating.
9. Use `radon cc` and `radon raw` (or the repository's configured Radon task)
   to identify complexity and line-count hotspots. Record exact commands and
   do not introduce Radon as a new dependency solely for one finding.

Classify each candidate as one of:

- `confirmed_dead`: no static, dynamic, public, generated, or known external consumer remains.
- `framework_contract`: retained for framework discovery or callback signatures.
- `compatibility_contract`: retained for a public or downstream consumer pending migration or deprecation.
- `protocol_signature`: retained to satisfy a typed or structural interface.
- `typing_only`: required only for static typing.
- `generated_or_migration`: generated or migration history that must not enter automatic deletion.
- `uncertain`: evidence is incomplete or conflicting.

Only the classifications supported by `AcceptedDeadCodeFinding` may be proposed for the YAML baseline. Keep other classifications in the inventory and explain the next evidence needed.

## Report inventory results

Write the report in German unless the user requests another language. Include:

- scan scope and explicit coverage gaps;
- command results and baseline state;
- candidate path and line, symbol or finding, tool confidence, classification, evidence, risk, and recommended action;
- separate sections for confirmed removals, retained contracts, and uncertain candidates;
- the narrowest verification required for each proposed removal cohort.

Do not describe a clean baseline guard as proof that no dead code exists. It proves only that the configured findings match the reviewed baseline.

## Remove code only when authorized

When removal is explicitly requested:

1. Select a small cohort of `confirmed_dead` candidates sharing one ownership boundary.
2. Remove related imports, re-exports, tests, configuration, and documentation only when their relationship is proven.
3. Preserve public contracts or define a documented deprecation and consumer-migration path before removal.
4. Keep workflow logic out of Django models and respect all clinical, storage, cryptographic, filesystem, and video invariants in `AGENTS.md`.
5. Update the reviewed baseline only when a removed or changed finding makes an entry stale, retaining schema-valid owner, reason, classification, and review data for exceptions.
6. For tracked feature status, use `feature-tracking/tracker.py update` or `verify --update`; never hand-edit assessment status.
7. For a behavior-preserving consolidation, keep the cohort small, preserve
   public names through an explicit compatibility path when required, and
   compare focused AAA tests, Pyright output, relevant coverage, and Radon
   metrics before and after. If callers cannot be ruled out, retain the code
   and classify it as `uncertain` or `compatibility_contract`.

## Verify changes

For code changes, run Pyright before pytest:

```bash
.devenv/state/venv/bin/pyright
.devenv/state/venv/bin/pytest <focused-path-or-nodeid>
devenv tasks run quality:dead-code
```

For a complexity-reduction cohort, the required evidence also includes:

- a shell-free, reproducible Radon measurement before and after;
- caller searches plus an isolated `python -m trace --count -C` run where
  runtime reachability is relevant;
- focused AAA tests covering the preserved behavior and invalid-input
  invariants; and
- Pyright before pytest, with coverage comparison when test coverage is in
  scope.

If a required tracker command cannot run because an unrelated feature YAML is
already invalid, do not repair or bypass that unrelated feature and do not
edit the requested scope without a lock. Report the exact validation error as
the blocker and preserve the working tree.

If the change crosses module boundaries, run the relevant broader repository task. Before a broad pytest lane, check `pgrep -af 'pytest|py.test'` and do not stack another suite on an unrelated run.

After changing a tracked feature, run:

```bash
./feature-tracking/tracker.py validate
./feature-tracking/tracker.py show code_quality
./feature-tracking/tracker.py check code_quality
```

Report what passed, what failed, unverified language or runtime surfaces, and the residual risk. Never claim a candidate is safely removable solely because Vulture, Ruff, Pyright, or a test suite does not reference it.
