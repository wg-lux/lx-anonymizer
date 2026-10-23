# Hierarchical agent context

Start at [`structure.yml`](../structure.yml). It links to the Python compute
package, Rust library, scripts, tests, documentation, feature assessments and
study evidence. The Python package links to frame cleaning, video processing,
report reading, OCR, detection, NER, language-model providers, anonymization,
training and evaluation. Small adapters, utilities, configuration and resource
folders remain covered by their parent; model artifacts, dataset descriptors,
Nix configuration and CI are indexed in the root inventory.

Follow only the branch relevant to the task, read its entrypoints and contracts,
then search inside that tree. Follow internal dependencies when an interface
crosses domains. Manifests describe navigation and ownership; they do not replace
code, `lx_dtypes` contracts, canonical specifications or existing feature YAML.
The bundled spaCy model manifest and feature assessments retain their own schemas.
`endoreg-db` owns durable attempts, leases, storage, publication and retries;
this repository produces unpublished processing candidates.

## Schema and relative paths

All manifests use version 1 and the common keys shown in the root manifest.
Unknown keys are allowed so domain ownership metadata can be preserved.
Responsibilities and boundaries must be nonempty lists; entrypoints, submodules
and contracts can be empty lists. Tiers are `root`, `domain`, `service`, `package`
and `tool`. Only the root has `parent: null`.

- `path` and `dependencies.internal` are relative to the repository root.
- `parent`, `entrypoints.file`, `submodules.path`, `submodules.manifest`,
  `contracts.doc`, `tests.directory` and optional `files` paths are relative to
  the containing manifest directory. `parent` points to an ancestor manifest.
- `files` accepts path strings or records with `path` and custom metadata such
  as `purpose`. It indexes significant files or directories without enumerating
  every helper. Any listed path must exist; directory and file fields enforce
  the corresponding kind. Absolute paths and paths escaping the repository,
  including through symlinks, fail validation.
- Each submodule names a descendant directory and its `structure.yml`. The
  child's parent must match the caller. Every discovered manifest must be
  reachable from the root, with exactly one caller. Manifest symlinks are rejected.
- Contract references are local files. Cross-repository obligations remain in
  AGENTS.md and the local concurrency contracts, rather than requiring another
  checkout to validate this repository. External dependencies are descriptive
  library or protocol names. Test commands run from repository root.

## Validation and maintenance

Run `python scripts/lint_structure.py` or `uv run python scripts/lint_structure.py`
from the development environment (PyYAML is an explicit dev dependency).
The linter is independent of application imports and uses Git's tracked plus
nonignored untracked files, catching new manifests before staging without
recursively loading media or model assets. Deleted tracked manifests fail;
ignored environments and build outputs are excluded. A linked ignored manifest
also fails, since it cannot be distributed as part of the context graph.
`--root PATH` supports another Git checkout, including validator test fixtures.

Validation checks schema types, references, bidirectional parent links and root
reachability, returns 0 on success, and reports errors with manifest paths on
stderr with exit code 1. It does not execute manifest test commands, infer
behavior from code, or automatically detect every unlisted helper addition.
Agents must maintain architectural descriptions and significant file inventories
in the same changeset as file additions, deletions, renames or boundary changes.

The linter runs through pre-commit (with an isolated PyYAML dependency), both
repository check scripts, `make lint-structure` and CI. For focused verification:

```sh
.devenv/state/venv/bin/pyright
uv sync --extra dev
uv run pytest tests/test_lint_structure.py -q
```

`make test-structure` performs this sequence after validating the graph.
The existing `scripts/run_checks.sh` provides the broader development lane.

## Delegated work

When delegation is authorized, give each subagent a manifest-owned directory,
the parent context, applicable contract references and its focused test command.
Keep edits within that tree; agree on shared interfaces and parent manifest edits
with the coordinating agent. A manifest's dependency edge permits reading a
neighbor's context, not changing its responsibilities without coordination.
New logical domains require a child manifest and a parent link in the same change;
trivial folders stay covered by the nearest manifest.
