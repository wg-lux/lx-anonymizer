#!/usr/bin/env python3
"""
collect_patient_names.py
────────────────────────
Usage:
    python collect_patient_names.py /path/to/folder [--llm-extractor <model>] [--out output.jsonl]

The script walks the folder non-recursively. Adapt Path.rglob("*.json") if
you need recursion. Writes results incrementally to a JSON Lines (.jsonl) file
and skips entries already present in the output file.

<model> can be 'deepseek', 'medllama', or 'llama3'.
If --llm-extractor is not provided, SpaCy/Regex extraction is used.
"""

from __future__ import annotations
import subprocess, time, logging, shutil, sys


import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
# Import tqdm
from tqdm import tqdm

from torch.utils.hipify.hipify_python import TrieNode

from .report_reader import ReportReader        # <- your class
from .custom_logger import get_logger

logger = get_logger(__name__)
# --------------------------------------------------------------------------- #
# helpers for JSON Lines handling
# --------------------------------------------------------------------------- #


def load_existing_ids_jsonl(output_file: Path) -> Set[Tuple[str, str]]:
    """Loads existing (file, report_id) tuples from a JSON Lines file."""
    existing_ids = set()
    if not output_file.exists():
        return existing_ids
    try:
        with output_file.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    file_id = data.get("file")
                    report_id = data.get("report_id")
                    if file_id and report_id:
                        existing_ids.add((str(file_id), str(report_id)))
                    else:
                        logger.warning(f"Skipping line {line_num+1} in {output_file}: Missing 'file' or 'report_id'.")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {line_num+1} in {output_file}: {line[:100]}...")
    except Exception as e:
        logger.error(f"Failed to read existing results from {output_file}: {e}")
        return set()
    logger.info(f"Loaded {len(existing_ids)} existing record identifiers from {output_file}")
    return existing_ids

def append_result_to_jsonl(output_file: Path, result: Dict):
    """Appends a single result dictionary as a JSON line to the output file."""
    try:
        json_string = json.dumps(result, ensure_ascii=False)
        with output_file.open('a', encoding='utf-8') as f:
            f.write(json_string + '\n')
    except Exception as e:
        logger.error(f"Failed to append result to {output_file}: {e} - Data: {result}")

# --------------------------------------------------------------------------- #
# Processing functions
# --------------------------------------------------------------------------- #

def calculate_pdf_hash(report_file_path: Path) -> Optional[str]:
    """Calculates the SHA256 hash of a PDF file."""
    # Construct the path to the corresponding PDF file
    pdf_file_path = report_file_path.with_suffix(".pdf")
    try:
        if not pdf_file_path.is_file():
            logger.error(f"Corresponding PDF file not found for {report_file_path.name}: {pdf_file_path}")
            return None
        hasher = hashlib.sha256()
        with pdf_file_path.open("rb") as pdf_file:
            while chunk := pdf_file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        # This specific error message was requested by the user prompt, keep it for clarity
        logger.error(f"Could not calculate PDF hash for {report_file_path.name}: [Errno 2] No such file or directory: '{pdf_file_path}'")
        return None
    except Exception as e:
        logger.error(f"Error calculating hash for {pdf_file_path}: {e}")
        return None

def process_report_file(report_file_path: Path, reader: ReportReader) -> List[Dict[str, Any]]:
    """Processes a single JSON report file and extracts patient names."""
    results = []
    try:
        with report_file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        # Calculate PDF hash using the full path
        pdf_hash = calculate_pdf_hash(report_file_path)

        if isinstance(data, list): # Handle cases where the JSON root is a list
            reports = data
        elif isinstance(data, dict) and "reports" in data: # Handle cases where reports are under a "reports" key
             reports = data["reports"]
        else:
             logger.warning(f"Unexpected JSON structure in {report_file_path.name}. Expected list or dict with 'reports' key.")
             return []


        for report_data in reports:
            report_id = report_data.get("id", "UnknownID") # Get report ID or use a default
            header_line = reader.find_header_line(report_data.get("text", ""))
            if header_line:
                patient_info = reader.extract_patient_data(header_line)
                if patient_info.get("patient_last_name") or patient_info.get("patient_first_name"):
                    results.append({
                        "file": report_file_path.name, # Store only the filename
                        "report_id": report_id,
                        "first_name": patient_info.get("patient_first_name"),
                        "last_name": patient_info.get("patient_last_name"),
                        # Add pdf_hash if needed in the output
                        # "pdf_hash": pdf_hash
                    })
            else:
                 logger.debug(f"No header line found in report {report_id} of file {report_file_path.name}")


    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {report_file_path}")
    except Exception as e:
        logger.error(f"Error processing file {report_file_path}: {e}")
    return results

def _process_json_file(
    rr: ReportReader,
    fp: Path,
    llm_extractor: str | None,
    existing_ids: Set[Tuple[str, str]],
    output_file: Path
) -> int:
    """
    Parse a JSON file, process reports not in existing_ids, append results to output_file.
    Returns the count of newly processed reports.
    """
    new_results_count = 0
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skipping %s (invalid JSON): %s", fp, exc)
        return 0

    if not isinstance(data, dict):
        logger.warning("Skipping %s: Expected top-level JSON object to be a dictionary, found %s", fp, type(data).__name__)
        return 0

    filename = fp.name
    for report_id, obj in data.items():
        current_id = (filename, str(report_id))
        if current_id in existing_ids:
            logger.debug(f"Skipping already processed report: {current_id}")
            continue

        if not isinstance(obj, dict):
            logger.warning(
                "Skipping report '%s' in file %s: Expected value to be a dictionary, found %s",
                report_id, filename, type(obj).__name__,
            )
            continue

        raw_text: str = obj.get("report", "")
        if not raw_text:
             logger.debug(
                "Skipping report '%s' in file %s: 'report' key missing or empty.",
                report_id, filename
            )
             continue

        extracted_meta = {}
        if llm_extractor:
            logger.info(f"Using LLM extractor '{llm_extractor}' for JSON report {report_id}")
            
            if llm_extractor == 'deepseek':

                extracted_meta = rr.extract_report_meta_deepseek(raw_text, fp)
            elif llm_extractor == 'medllama':
                extracted_meta = rr.extract_report_meta_medllama(raw_text, fp)
            elif llm_extractor == 'llama3':
                extracted_meta = rr.extract_report_meta_llama3(raw_text, fp)
            else:
                 logger.warning(f"Unknown LLM extractor '{llm_extractor}' specified for JSON. Falling back.")
                 extracted_meta = rr.extract_report_meta(raw_text, fp)

            if not extracted_meta:
                 logger.warning(f"LLM extractor '{llm_extractor}' failed for JSON report {report_id}. Falling back to SpaCy/Regex.")
                 extracted_meta = rr.extract_report_meta(raw_text, fp)
        else:
            logger.info(f"Using default SpaCy/Regex extractor for JSON report {report_id}")
            extracted_meta = rr.extract_report_meta(raw_text, fp)

        result_obj = {
            "file": filename,
            "report_id": str(report_id),
            "first_name": extracted_meta.get("patient_first_name"),
            "last_name": extracted_meta.get("patient_last_name"),
            "dob": str(extracted_meta.get("patient_dob")) if extracted_meta.get("patient_dob") else None,
            "gender": extracted_meta.get("patient_gender"),
            "casenumber": extracted_meta.get("casenumber"),
        }

        append_result_to_jsonl(output_file, result_obj)
        existing_ids.add(current_id)
        new_results_count += 1
        logger.debug(f"Appended result for {current_id}")

    return new_results_count


def _process_pdf_file(
    rr: ReportReader,
    fp: Path,
    llm_extractor: str | None,
    existing_ids: Set[Tuple[str, str]],
    output_file: Path
) -> int:
    """
    Process a PDF if not in existing_ids, append result to output_file.
    Returns 1 if processed, 0 otherwise.
    """
    filename = fp.name
    report_id = fp.stem
    current_id = (filename, str(report_id))

    if current_id in existing_ids:
        logger.debug(f"Skipping already processed PDF: {current_id}")
        return 0

    original_text, anonymized_text, extracted_meta = rr.process_report(
        pdf_path=fp,
        use_llm_extractor=llm_extractor
    )

    if not extracted_meta:
        logger.warning(f"Metadata extraction failed for PDF {filename}. Skipping append.")
        return 0

    result_obj = {
        "file": filename,
        "report_id": str(report_id),
        "first_name": extracted_meta.get("patient_first_name"),
        "last_name": extracted_meta.get("patient_last_name"),
        "dob": str(extracted_meta.get("patient_dob")) if extracted_meta.get("patient_dob") else None,
        "gender": extracted_meta.get("patient_gender"),
        "casenumber": extracted_meta.get("casenumber"),
    }

    append_result_to_jsonl(output_file, result_obj)
    existing_ids.add(current_id)
    logger.debug(f"Appended result for {current_id}")
    return 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def collect_names(folder: Path, output_file: Path, llm_extractor: str | None) -> None:
    """Collect names using the specified extraction method, writing incrementally."""
    rr = ReportReader()
    existing_ids = load_existing_ids_jsonl(output_file)
    total_new_results = 0

    logger.info(f"Starting name collection from folder: {folder}")
    logger.info(f"Output will be appended to: {output_file}")
    if llm_extractor:
        logger.info(f"Using LLM extractor: {llm_extractor}")
    else:
        logger.info("Using default SpaCy/Regex extractor.")

    # Get list of files to process for tqdm
    files_to_process = [fp for fp in folder.iterdir() if fp.is_file()]

    # Wrap the loop with tqdm for progress bar
    for fp in tqdm(files_to_process, desc="Processing files", unit="file"):
        newly_added_count = 0
        try:
            if fp.suffix.lower() == ".json":
                newly_added_count = _process_json_file(rr, fp, llm_extractor, existing_ids, output_file)
            elif fp.suffix.lower() == ".pdf":
                newly_added_count = _process_pdf_file(rr, fp, llm_extractor, existing_ids, output_file)
            else:
                logger.debug("Ignoring unsupported file type: %s", fp)

            if newly_added_count > 0:
                total_new_results += newly_added_count

        except Exception as e:
            logger.error(f"Failed to process file {fp.name}: {e}", exc_info=True)

    logger.info(f"Finished processing folder. Added {total_new_results} new records in total to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect patient names from JSON/PDF reports using different extraction methods. Writes incrementally to JSON Lines (.jsonl).")
    parser.add_argument("folder", type=Path, help="Folder with JSON/PDF reports")
    parser.add_argument(
        "--llm-extractor",
        type=str,
        choices=['deepseek', 'medllama', 'llama3'],
        default=None,
        help="Specify the LLM extractor model to use (e.g., 'deepseek', 'medllama', 'llama3'). If not provided, uses SpaCy/Regex.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("extracted_names.jsonl"),
        help="Output JSON Lines file (default: extracted_names.jsonl)",
    )
    args = parser.parse_args()

    collect_names(args.folder, args.out, args.llm_extractor)
