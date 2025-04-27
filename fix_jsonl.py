#!/usr/bin/env python3
"""
fix_jsonl.py - Reads a potentially broken JSON or JSONL file, cleans it,
               and writes the output as a valid JSON Lines (.jsonl) file.

Usage:
    python fix_jsonl.py <input_file> <output_file.jsonl>

Attempts to:
- Remove single-line (//) and multi-line (/* ... */) comments.
- Handle files containing a single JSON list `[...]` by outputting each element as a line.
- Handle files containing multiple JSON objects separated by newlines (or commas).
- Handle trailing commas in lists and objects.
"""

import json
import re
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def remove_comments(text: str) -> str:
    """Removes // and /* ... */ comments from a string."""
    # Remove /* ... */ comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove // comments
    text = re.sub(r'//.*', '', text)
    return text

def clean_json_text(text: str) -> str:
    """Applies various cleaning steps to make JSON text more parsable."""
    text = remove_comments(text)
    # Attempt to remove trailing commas within objects and arrays
    # This is a common issue, especially when files are manually edited.
    # Be careful, this regex might be too aggressive in some edge cases.
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text.strip()

def objects_from_text(text: str):
    """
    Yields JSON objects found in the text.
    Tries parsing as a single list first, then line-by-line.
    Handles optional trailing commas on lines.
    """
    try:
        # Try parsing the whole text as a single JSON object or list
        data = json.loads(text)
        if isinstance(data, list):
            # If it's a list, yield each item
            logger.info("Input parsed as a JSON list. Yielding elements.")
            yield from data
            return # Successfully processed as a list
        elif isinstance(data, dict):
            # If it's a single object, yield it
            logger.info("Input parsed as a single JSON object.")
            yield data
            return # Successfully processed as an object
        else:
            logger.warning(f"Parsed root JSON is neither list nor dict (type: {type(data)}). Skipping.")
            return

    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse input as single JSON: {e}. Trying line-by-line parsing.")
        # If parsing as a single object failed, try line by line (typical for JSONL)
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # Remove trailing comma if present before parsing
            if line.endswith(','):
                line = line[:-1]
            # Re-strip in case the comma removal left whitespace
            line = line.strip()
            if not line: # Skip if line becomes empty after removing comma
                continue
            try:
                # Try parsing each non-empty line
                yield json.loads(line)
            except json.JSONDecodeError as json_err: # Specific exception
                logger.warning(f"Skipping invalid JSON on line {i+1}: {json_err}. Content: {line[:100]}...")
            except Exception as general_err: # Catch other potential errors
                 logger.error(f"Unexpected error processing line {i+1}: {general_err}. Content: {line[:100]}...")

def main(input_path_str: str, output_path_str: str):
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if output_path.suffix.lower() != ".jsonl":
        logger.warning(f"Output file '{output_path}' does not have .jsonl extension.")

    try:
        raw_text = input_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to read input file {input_path}: {e}")
        sys.exit(1)

    cleaned_text = clean_json_text(raw_text)

    count = 0
    try:
        with output_path.open('w', encoding='utf-8') as outfile:
            for obj in objects_from_text(cleaned_text):
                if isinstance(obj, dict): # Ensure we are writing objects
                    json.dump(obj, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    count += 1
                else:
                    logger.warning(f"Skipping non-object item found: {type(obj)}")
        logger.info(f"Successfully wrote {count} objects to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write to output file {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_jsonl.py <input_file> <output_file.jsonl>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
