import json
import re  # Import regex
from ollama import chat, ResponseError, RequestError
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from pydantic import ValidationError
from .schema import PatientMeta
from .custom_logger import get_logger
from .spacy_regex import PatientDataExtractorLg

logger = get_logger(__name__)

''' This module provides functionality to extract patient metadata from medical reports using Ollama's structured output feature. It defines prompts, schemas, and a function to handle the extraction process with retries and error handling. '''

# Prompt templates for structured extraction
SYSTEM_PROMPT = (
    "You are an expert medical report information extractor.\n"
    "Your task is to extract specific patient metadata based on the provided text.\n"
    "You MUST return ONLY a single, valid JSON object matching the following schema. Do NOT include any other text, explanations, comments, markdown formatting, or introductory sentences before or after the JSON object.\n"
    "Schema:\n"
    "{schema}\n"
    "If a value is not found or cannot be determined, use `null` for that field in the JSON."
)

USER_PROMPT_TEMPLATE = 'Extract patient metadata from the following report:\n"""{report}"""'

PATIENT_META_SCHEMA = PatientMeta.model_json_schema()

PatientDataExtractorLg = PatientDataExtractorLg()

def _extract_json_block(text: str) -> str | None:
    """Tries to extract the first valid JSON block {} from a string."""
    # Find the first '{' and the last '}'
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index != -1 and end_index != -1 and start_index < end_index:
        potential_json = text[start_index: end_index + 1]
        try:
            # Try to parse it to ensure it's valid JSON
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            logger.warning("Found braces, but content was not valid JSON.")
            # Fallback: Regex to find JSON object (less robust)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                potential_json_re = match.group(0)
                try:
                    json.loads(potential_json_re)
                    logger.debug("Extracted JSON block using regex fallback.")
                    return potential_json_re
                except json.JSONDecodeError:
                    logger.warning("Regex match was not valid JSON either.")
                    return None
            else:
                return None  # No JSON object found
    else:
        logger.warning("Could not find enclosing braces {} in the response.")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def extract_meta_ollama(text: str, model: str = "llama3:8b") -> dict:
    """
    Extracts patient metadata using Ollama's structured output feature.

    Args:
        text: The report text to extract metadata from.
        model: The Ollama model name to use (e.g., "deepseek-r1:1.5b", "llama3:8b").

    Returns:
        A dictionary containing the extracted metadata, validated against PatientMeta.

    Raises:
        RetryError: If extraction fails after multiple retries.
        Exception: For other unexpected errors during the process.
    """
    logger.info(f"Attempting metadata extraction with Ollama model: {model}")
    raw_response_content = ""  # Initialize to handle potential errors before assignment
    cleaned_json_str = None   # Initialize cleaned_json_str to avoid UnboundLocalError
    
    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(schema=json.dumps(PATIENT_META_SCHEMA, indent=2))
                },
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(report=text)
                },
            ],
            format="json",
            options={"temperature": 0.0},
        )

        raw_response_content = response['message']['content']
        logger.debug(f"Raw Ollama response content:\n{raw_response_content}")
        
        # Extract JSON block from response
        cleaned_json_str = _extract_json_block(raw_response_content)
        
        if cleaned_json_str:
            try:
                # Try to validate with Pydantic first
                validated_data = PatientMeta.model_validate_json(cleaned_json_str)
                logger.info(f"Successfully extracted and validated metadata with {model}")
                return validated_data.model_dump(mode='json')
            except ValidationError as e:
                logger.warning(f"Pydantic validation failed: {e}. Falling back to regex extraction.")
                # Fall through to regex fallback
        
        # Fallback: Use regex-based extraction
        logger.info("Using regex fallback for metadata extraction")
        meta = PatientDataExtractorLg.regex_extract_llm_meta(raw_response_content)
        
        # Return the regex-extracted metadata
        logger.info(f"Regex extraction completed with {model}")
        return meta

    except ResponseError as e:
        logger.error(f"Ollama API Response Error (model: {model}): {e.status_code} - {e.error}")
        raise
    except RequestError as e:
        logger.error(f"Ollama API Request Error (model: {model}): {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response (model: {model}): {e}")
        logger.error(f"Content attempted for parsing: {cleaned_json_str or 'N/A'}")
        raise
    except Exception as e:
        logger.error(f"Error during Ollama extraction or validation (model: {model}): {e}")
        if cleaned_json_str:
            logger.error(f"Cleaned content causing validation error: {cleaned_json_str}")
        elif raw_response_content:
            logger.error(f"Raw content (no JSON block found or cleaning failed): {raw_response_content}")
        raise  # Reraise to trigger tenacity retry


# handle the final failure after retries
def extract_with_fallback(text: str, model: str) -> dict:
    """Wrapper to call extract_meta_ollama and handle final RetryError."""
    try:
        return extract_meta_ollama(text, model=model)
    except RetryError as e:
        logger.error(f"Ollama extraction failed permanently after multiple retries for model {model}: {e}")
        return {}  # Return empty dict to signal failure to the caller
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ollama extraction for model {model}: {e}")
        return {}  # Return empty dict on other unexpected errors
