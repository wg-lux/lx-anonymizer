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

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _process_json_file(rr: ReportReader, fp: Path) -> List[Dict[str, str]]:
    """Parse a single JSON file with the synthetic-report layout."""
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        rr.logger.warning("Skipping %s (invalid JSON): %s", fp, exc)
        return []

    results: List[Dict[str, str]] = []
    for report_id, obj in data.items():
        raw_text: str = obj.get("report", "")
        extracted = rr.patient_extractor(raw_text)
        results.append(
            {
                "file": str(fp.name),
                "report_id": report_id,
                "first_name": extracted.get("patient_first_name"),
                "last_name": extracted.get("patient_last_name"),
            }
        )
    return results


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
            rr.logger.debug("Ignoring unsupported file type: %s", fp)

    # write results
    output_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    rr.logger.info("Saved %d records → %s", len(all_results), output_file)


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
