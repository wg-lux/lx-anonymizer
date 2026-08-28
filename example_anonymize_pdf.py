#!/usr/bin/env python3
"""
Example script showing how to use ReportReader to create anonymized PDFs
with sensitive regions blackened out.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from lx_anonymizer.report_contracts import (
    ReportAnonymizationOptions,
    ReportAnonymizationRequest,
)
from lx_anonymizer.report_reader import ReportReader


def _request(
    source_path: Path,
    output_directory: Path,
    *,
    use_llm: bool | None = None,
) -> ReportAnonymizationRequest:
    source_bytes = source_path.read_bytes()
    output_directory.mkdir(parents=True, exist_ok=True)
    return ReportAnonymizationRequest(
        attempt_id=uuid4(),
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_size_bytes=len(source_bytes),
        output_directory=output_directory,
        options=ReportAnonymizationOptions(use_llm=use_llm),
    )


def example_pdf_anonymization() -> None:
    """Process one immutable PDF into an attempt-owned anonymized artifact."""
    print("\n" + "=" * 60)
    print("Example 2: PDF with Blackened Sensitive Regions")
    print("=" * 60)

    reader = ReportReader()

    result = reader.process_report(
        _request(
            Path("path/to/report.pdf"),
            Path("output/attempt"),
        )
    )

    print(f"Extracted metadata: {result.extracted_metadata.model_fields_set}")
    print(f"Anonymized PDF created: {result.artifact_path}")
    print(f"Artifact SHA-256: {result.artifact_sha256}")


def example_with_cropping() -> None:
    """Example: Advanced processing with region cropping"""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Processing with Cropping")
    print("=" * 60)

    reader = ReportReader()

    # Use the advanced method with cropping
    original_text, anonymized_text, metadata, cropped_info, anonymized_pdf = reader.process_report_with_cropping(
        pdf_path="path/to/report.pdf",
        crop_sensitive_regions=True,  # Extract sensitive regions as separate images
        crop_output_dir="output/cropped_regions/",
        anonymization_output_dir="output/anonymized/",
    )

    print(f"Extracted metadata: {metadata.keys()}")
    print(f"Cropped regions: {metadata.get('total_cropped_regions', 0)}")
    print(f"Anonymized PDF: {anonymized_pdf}")


def example_llm_extraction() -> None:
    """Example: Use LLM for enhanced metadata extraction"""
    print("\n" + "=" * 60)
    print("Example 4: LLM-Enhanced Extraction with PDF Masking")
    print("=" * 60)

    reader = ReportReader()

    result = reader.process_report(
        _request(
            Path("path/to/report.pdf"),
            Path("output/llm-attempt"),
            use_llm=True,
        )
    )

    print(f"LLM available: {reader.llm_available}")
    print(f"Ollama available: {reader.ollama_available}")
    print(f"Extracted metadata: {result.extracted_metadata.model_fields_set}")
    print(f"Anonymized PDF: {result.artifact_path}")


if __name__ == "__main__":
    print("ReportReader PDF Anonymization Examples")
    print("=" * 60)
    print("\nThese examples show different ways to anonymize medical reports:")
    print("1. Canonical PDF anonymization")
    print("2. Advanced region cropping (for analysis)")
    print("3. LLM-enhanced extraction (if a provider is available)")
    print("\n" + "=" * 60)

    # Uncomment to run examples:
    # example_pdf_anonymization()
    # example_with_cropping()
    # example_llm_extraction()

    print("\n✅ To use: Uncomment the example functions above and provide real PDF paths")
