#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
base_model="${OLLAMA_GEMMA4_BASE_MODEL:-gemma4:e2b}"
runtime_model="${LLM_MODEL:-lx-gemma4-e2b-json}"
ollama_url="${OLLAMA_URL:-http://127.0.0.1:11434}"
ollama_host="${OLLAMA_CLIENT_HOST:-127.0.0.1:11434}"
modelfile="${OLLAMA_GEMMA4_MODELFILE:-$repo_root/ollama/Modelfile.gemma4-ocr}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is required but is not installed on this machine." >&2
  exit 1
fi

if [[ ! -f "$modelfile" ]]; then
  echo "Gemma 4 Modelfile not found: $modelfile" >&2
  exit 1
fi

if ! curl --fail --silent --show-error --max-time 2 "$ollama_url/api/tags" >/dev/null 2>&1; then
  OLLAMA_HOST="$ollama_host" nohup ollama serve >"${TMPDIR:-/tmp}/lx-ollama.log" 2>&1 &
fi

for _ in $(seq 1 30); do
  if curl --fail --silent --show-error --max-time 2 "$ollama_url/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl --fail --silent --show-error --max-time 2 "$ollama_url/api/tags" >/dev/null
OLLAMA_HOST="$ollama_host" ollama pull "$base_model"
OLLAMA_HOST="$ollama_host" ollama create "$runtime_model" --file "$modelfile"
OLLAMA_HOST="$ollama_host" ollama show "$runtime_model" >/dev/null

echo "Ollama model $runtime_model is ready at $ollama_url"
