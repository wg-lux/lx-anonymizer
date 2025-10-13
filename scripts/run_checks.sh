#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# Ensure dependencies are in sync
if ! command -v uv >/dev/null 2>&1; then
  echo "[run_checks] Installing uv (requires curl)" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra dev --extra llm --extra ocr

# Run linting
uv run flake8

# Run CPU-friendly test suite
uv run pytest -m "not gpu" --maxfail=1 --disable-warnings "$@"

echo "All checks passed."
