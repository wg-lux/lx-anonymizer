#!/usr/bin/env python3
"""
collect_patient_names.py
────────────────────────
Usage:
    python collect_patient_names.py /path/to/folder  [--out output.json]

The script walks the folder non-recursively.  Adapt Path.rglob("*.json") if
you need recursion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .report_reader import ReportReader        # <- your class
from .custom_logger import get_logger
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

logger = get_logger(__name__)

def _process_json_file(rr: ReportReader, fp: Path) -> List[Dict[str, str]]:
    """Return patient-name rows or an empty list if structure not as expected."""
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skipping %s (invalid JSON): %s", fp, exc)
        return []

    # ── expect a dict at the top level ───────────────────────────────────────
    if not isinstance(data, dict):
        logger.info("Skip %s: root JSON is %s, not an object.", fp, type(data).__name__)
        return []

    rows: List[Dict[str, str]] = []
    for report_id, obj in data.items():
        if not isinstance(obj, dict):
            logger.info("Skip %s › %s: entry is %s, expected dict.", fp.name, report_id, type(obj).__name__)
            continue

        raw_text = obj.get("report")
        if not isinstance(raw_text, str) or not raw_text.strip():
            logger.info("Skip %s › %s: no usable 'report' field.", fp.name, report_id)
            continue

        extracted = rr.extract_report_meta(raw_text, fp.name)
        rows.append(
            {
                "file": fp.name,
                "report_id": report_id,
                "first_name": extracted.get("patient_first_name"),
                "last_name": extracted.get("patient_last_name"),
            }
        )
    return rows



def _process_pdf_file(rr: ReportReader, fp: Path) -> Dict[str, str]:
    """Extract patient name(s) from a single PDF."""
    text = rr.read_pdf(fp)
    extracted = rr.patient_extractor(text)
    return {
        "file": str(fp.name),
        "report_id": fp.stem,
        "first_name": extracted.get("patient_first_name"),
        "last_name": extracted.get("patient_last_name"),
    }
    
    


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def collect_names(folder: Path, output_file: Path) -> None:
    rr = ReportReader()  # default locale/extractors
    all_results: List[Dict[str, str]] = []

    for fp in folder.iterdir():  # non-recursive; switch to rglob for deep walk
        if not fp.is_file():
            continue

        if fp.suffix.lower() == ".json":
            all_results.extend(_process_json_file(rr, fp))

        elif fp.suffix.lower() == ".pdf":
            all_results.append(_process_pdf_file(rr, fp))

        else:
            logger.debug("Ignoring unsupported file type: %s", fp)

    # write results
    output_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    logger.info("Saved %d records → %s", len(all_results), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder with JSON/PDF reports")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("extracted_names.json"),
        help="Output JSON file (default: extracted_names.json)",
    )
    args = parser.parse_args()

    collect_names(args.folder, args.out)
