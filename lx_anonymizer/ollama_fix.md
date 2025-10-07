Here’s a tight, execution-ready **agent implementation prompt** you can drop into your tooling. It encodes the decisions you made and guards against “verschlimmbessern” by setting strict contracts, tests, and review gates.


# Agent Implementation Prompt — Fix Pydantic/Regex Fallback Bug

## Objective

Fix the metadata extraction function so that:

* It **always returns a `dict`** (JSON-serializable).
* It **returns Pydantic-validated JSON** immediately on success.
* It **only** uses regex fallback when Pydantic validation **fails** (including JSON decode errors).
* It **never** reassigns a Pydantic model variable to a `dict`.
* Logs are PHI-safe and clearly indicate the path: `"pydantic_success"` vs `"regex_fallback"`.

## Scope

* Language: Python
* Affected module(s): the LLM metadata extraction flow that currently contains:

```python
try:
    validated_data = PatientMeta.model_validate_json(cleaned_json_str)
    logger.info(f"Successfully extracted and validated metadata with {model}")
    logger.debug(f"Extracted metadata: {validated_data.model_dump(mode='json')}")
    logger.debug(f"Regex-extracted validated metadata used for llm extraction: {meta}")

    validated_data = PatientDataExtractorLg.regex_extract_llm_meta(cleaned_json_str, raw_response_content)
    meta = validated_data.model_dump(mode='json')
except ValidationError as e:
    ...
```

> Note: The exact file/function name may vary. Search for this snippet and patch **in place**.

## Contracts & Behavior (Authoritative)

1. **Return Type:** `dict` only.
2. **Primary Path:** If `PatientMeta.model_validate_json(cleaned_json_str)` succeeds, **return** `validated.model_dump(mode="json")` immediately.
3. **Fallback Path:** If Pydantic validation or JSON parsing fails, call
   `PatientDataExtractorLg.regex_extract_llm_meta(cleaned_json_str, raw_response_content)` and **return its `dict`** (or `{}` on error).
4. **No Merge:** Do **not** merge regex data into Pydantic-validated data.
5. **No Reassignment Bug:** Do **not** overwrite `validated_data` (a Pydantic model) with a `dict`.
6. **Error Handling:** Catch `pydantic.ValidationError` **and** `json.JSONDecodeError` (or `ValueError` if library-agnostic) for the primary path. For regex path, catch generic `Exception`, log, and return `{}`.
7. **Logging (PHI-safe):**

   * `INFO`: One-line outcome: `"pydantic_success"` or `"regex_fallback"`, include `model` name if available.
   * `DEBUG`: Log **keys-only** or a **redacted** subset. Never log full PHI content.
   * Fix typo: `"ontent"` → `"Content"`.
8. **Regex Contract:** `regex_extract_llm_meta` returns `dict[str, str | None]`. Treat as final output for fallback path.

## Implementation Steps

1. Locate the function containing the provided snippet.
2. Replace the try/except block with the **Corrected Implementation** below.
3. Ensure imports include `json` and `ValidationError` from Pydantic v2 API.
4. Add a small PHI-safe utility to log keys (or allow-list) if not present already.

## Corrected Implementation (drop-in)

```python
import json
from pydantic import ValidationError

# Optional: minimal PHI-safe logging helper
def _safe_keys_view(d: dict | None) -> list[str]:
    try:
        return sorted(list(d.keys())) if isinstance(d, dict) else []
    except Exception:
        return []

def extract_patient_meta(cleaned_json_str: str | None,
                         raw_response_content: str | None,
                         model: str = "unknown-llm") -> dict:
    """
    Returns metadata as a JSON-serializable dict.
    Prefers Pydantic-validated JSON; falls back to regex extraction.
    """

    # Primary path: Pydantic validation of structured JSON
    try:
        if not cleaned_json_str:
            raise ValueError("No cleaned_json_str provided")

        validated_model = PatientMeta.model_validate_json(cleaned_json_str)
        result: dict = validated_model.model_dump(mode="json")

        logger.info("metadata_extraction_path=pydantic_success model=%s", model)
        logger.debug("metadata_keys_pydantic=%s", _safe_keys_view(result))
        return result

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Pydantic/JSON validation failed: %s; falling back to regex. model=%s",
                       str(e), model)

    # Fallback path: regex extraction
    try:
        meta: dict = PatientDataExtractorLg.regex_extract_llm_meta(
            cleaned_json_str or "",
            raw_response_content or "",
        ) or {}

        logger.info("metadata_extraction_path=regex_fallback model=%s", model)
        logger.debug("Content used for regex extraction available=%s raw_available=%s",
                     bool(cleaned_json_str), bool(raw_response_content))
        logger.debug("metadata_keys_regex=%s", _safe_keys_view(meta))
        return meta

    except Exception as rex:
        logger.error("Regex extraction failed: %s; returning empty metadata. model=%s", str(rex), model)
        return {}
```

> If your function name/signature differs, adapt the name but preserve the **behavioral contract** and **return type**.

## Tests (must add / update)

Create or update unit tests to enforce behavior:

1. **test_pydantic_success_returns_validated_dict**

   * Given valid `cleaned_json_str`, assert:

     * returns a `dict`
     * does **not** call regex
     * logs contain `metadata_extraction_path=pydantic_success`

2. **test_pydantic_validation_error_falls_back_to_regex**

   * Given invalid JSON / schema, assert:

     * regex is called once
     * returns the dict from regex
     * logs contain `metadata_extraction_path=regex_fallback`

3. **test_no_merge_occurs**

   * Spy on regex and Pydantic:

     * when Pydantic succeeds, regex is **not** called.

4. **test_regex_failure_returns_empty_dict**

   * Make regex raise; assert `{}` returned and error log produced.

5. **test_logs_are_phi_safe**

   * Ensure debug logs do **not** include raw values from the metadata (only keys).

> Use dependency injection or monkeypatching to spy on `PatientMeta.model_validate_json` and `PatientDataExtractorLg.regex_extract_llm_meta`. If a global instance is used, monkeypatch the method.

## Acceptance Criteria (DoD)

* All tests above **green**.
* Grep shows **no** occurrences of `validated_data = PatientDataExtractorLg.regex_extract_llm_meta(...)` or similar reassignment patterns.
* Function **always** returns `dict`.
* Logs show **exact** markers:

  * `metadata_extraction_path=pydantic_success`
  * `metadata_extraction_path=regex_fallback`
* Typo fixed: “Content used for regex extraction”.
* No PHI leaks in debug logs (keys-only or allow-listed fields).

## Non-goals

* No schema changes to `PatientMeta`.
* No merging policy; strictly either Pydantic or regex.
* No refactors outside the immediate function unless required by imports or tests.

---

If the codebase uses different names, apply a search-and-replace carefully but **do not** alter the contract.
