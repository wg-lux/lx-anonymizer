#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

echo "[pre-commit] Checking repository state..."

if [[ -f ".git/MERGE_HEAD" ]]; then
  echo "Error: merge in progress (MERGE_HEAD exists)."
  exit 1
fi

if [[ -d ".git/rebase-apply" || -d ".git/rebase-merge" ]]; then
  echo "Error: rebase in progress."
  exit 1
fi

if [[ -f ".git/CHERRY_PICK_HEAD" || -f ".git/REVERT_HEAD" ]]; then
  echo "Error: cherry-pick or revert in progress."
  exit 1
fi

if [[ -n "$(git ls-files -u)" ]]; then
  echo "Error: unresolved merge conflicts detected."
  exit 1
fi

# Detect conflict markers/whitespace errors in staged changes.
if ! git diff --cached --check --quiet; then
  echo "Error: staged diff contains conflict markers or whitespace issues."
  exit 1
fi

BEFORE_STATUS="$(git status --porcelain=v1)"

echo "[pre-commit] Running test suite..."
pytest -q

AFTER_STATUS="$(git status --porcelain=v1)"
if [[ "$BEFORE_STATUS" != "$AFTER_STATUS" ]]; then
  echo "Error: repository state changed while running tests."
  echo "Run 'git status --short' and clean up generated changes."
  exit 1
fi

echo "[pre-commit] Checks passed."
