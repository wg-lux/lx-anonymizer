#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.devenv/state/venv}"
UV_BIN="${UV_BIN:-uv}"

# Ensure dependencies are in sync
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [ -x ".devenv/state/venv/bin/uv" ]; then
    UV_BIN=".devenv/state/venv/bin/uv"
  elif [ -x ".venv/bin/uv" ]; then
    UV_BIN=".venv/bin/uv"
  fi
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "[run_checks] Installing uv (requires curl)" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
  UV_BIN="${HOME}/.local/bin/uv"
fi

"$UV_BIN" sync --extra dev --extra cpu

# Run strict type checking before test execution
.devenv/state/venv/bin/pyright

# Run linting
"$UV_BIN" run flake8

# Run CPU-friendly test suite
"$UV_BIN" run pytest -m "not gpu" --maxfail=1 --disable-warnings "$@"

echo "All checks passed."
