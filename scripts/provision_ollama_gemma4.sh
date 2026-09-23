#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
base_model="${OLLAMA_GEMMA4_BASE_MODEL:-gemma4:e2b}"
runtime_model="${LLM_MODEL:-lx-gemma4-e2b-json}"
# Use a single endpoint for discovery and every CLI operation. Service lifecycle
# belongs to systemd, Docker, launchd or the operator, not this provisioning helper.
export OLLAMA_HOST="${LLM_BASE_URL:-${OLLAMA_URL:-${OLLAMA_CLIENT_HOST:-${OLLAMA_HOST:-http://127.0.0.1:11434}}}}"
modelfile="${OLLAMA_GEMMA4_MODELFILE:-$repo_root/ollama/Modelfile.gemma4-ocr}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is required but is not installed on this machine." >&2
  exit 1
fi

if [[ ! -f "$modelfile" ]]; then
  echo "Gemma 4 Modelfile not found: $modelfile" >&2
  exit 1
fi

if ! ollama list >/dev/null; then
  echo "Configured Ollama server is unavailable. Start its managed service before provisioning." >&2
  exit 1
fi
ollama pull "$base_model"
ollama create "$runtime_model" --file "$modelfile"
ollama show "$runtime_model" >/dev/null

echo "Ollama model $runtime_model is ready."
