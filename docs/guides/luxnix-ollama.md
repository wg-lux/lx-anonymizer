# LuxNix Ollama models

The LuxNix Ollama module provisions `gemma4:e2b` and the derived
`lx-gemma4-e2b-json` profile when `enableModelBootstrap` is enabled. Its default
is disabled; enabling Ollama alone does not install models.

Use these process settings for lx-anonymizer:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=lx-gemma4-e2b-json
LLM_OCR_MODEL=gemma4:e2b
```

Metadata extraction uses the JSON profile, whose system prompt permits only the
five metadata keys and whose generation limit is 256 tokens. OCR transcription
and correction use the base model so they do not inherit those restrictions.
Metadata requests for `lx-gemma4-e2b-json` (including its `:latest` alias)
explicitly disable thinking so the 256-token budget can produce the answer.
See [Ollama thinking controls](https://docs.ollama.com/capabilities/thinking).

An empty `LLM_OCR_MODEL` selects `gemma4:e2b` for Ollama and `LLM_MODEL` for
OpenAI-compatible providers. Explicit model overrides remain supported.

The local `scripts/provision_ollama_gemma4.sh` helper creates the same JSON profile
from `ollama/Modelfile.gemma4-json`. On LuxNix, let NixOS own model provisioning.
The LuxNix service also defaults to a one-minute keep-alive, one loaded model,
and one parallel request. These are server settings; client `.env` settings do
not reconfigure an already running systemd service.

Configuration is read at import time. Restart consuming web and worker processes
after changing their environment. A log selecting `local-model` indicates an
explicit override or a synthetic test configuration, not this project's default.
Check the environment of the actual consumer, including its working-directory
`.env`, and verify `/api/tags` lists `lx-gemma4-e2b-json:latest` and `gemma4:e2b`.
HTTP 503 alone does not establish a model-name problem; inspect the Ollama service
logs if inference still fails with the correct model. No automatic substitution
of an unavailable metadata model is performed.
