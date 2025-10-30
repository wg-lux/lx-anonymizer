#!/usr/bin/env python3
"""
Example script showing how to use ReportReader to create anonymized PDFs
with sensitive regions blackened out.
"""

from lx_anonymizer.report_reader import ReportReader
from pathlib import Path

def example_basic_anonymization():
    """Example: Process report with text anonymization only (no PDF masking)"""
    print("=" * 60)
    print("Example 1: Text Anonymization Only")
    print("=" * 60)
    
    reader = ReportReader()
    
    # Process report - returns text only
    original_text, anonymized_text, metadata, anonymized_pdf = reader.process_report(
        pdf_path="path/to/report.pdf",
        create_anonymized_pdf=False  # Default: no PDF masking
    )
    
    print(f"Extracted metadata: {metadata.keys()}")
    print(f"Anonymized PDF created: {anonymized_pdf is not None}")
    

def example_pdf_with_masking():
    """Example: Process report and create PDF with sensitive regions blackened"""
    print("\n" + "=" * 60)
    print("Example 2: PDF with Blackened Sensitive Regions")
    print("=" * 60)
    
    reader = ReportReader()
    
    # Process report and create anonymized PDF
    original_text, anonymized_text, metadata, anonymized_pdf = reader.process_report(
        pdf_path="path/to/report.pdf",
        create_anonymized_pdf=True,  # Enable PDF masking!
        anonymized_pdf_output_path="output/report_anonymized.pdf"  # Optional: custom path
    )
    
    print(f"Extracted metadata: {metadata.keys()}")
    print(f"Anonymized PDF created: {anonymized_pdf}")
    print(f"Anonymized PDF path in metadata: {metadata.get('anonymized_pdf_path')}")


def example_with_cropping():
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
        anonymization_output_dir="output/anonymized/"
    )
    
    print(f"Extracted metadata: {metadata.keys()}")
    print(f"Cropped regions: {metadata.get('total_cropped_regions', 0)}")
    print(f"Anonymized PDF: {anonymized_pdf}")


def example_llm_extraction():
    """Example: Use LLM for enhanced metadata extraction"""
    print("\n" + "=" * 60)
    print("Example 4: LLM-Enhanced Extraction with PDF Masking")
    print("=" * 60)
    
    reader = ReportReader()
    
    # Use LLM for better metadata extraction + create masked PDF
    original_text, anonymized_text, metadata, anonymized_pdf = reader.process_report(
        pdf_path="path/to/report.pdf",
        use_llm_extractor='deepseek',  # Use LLM for metadata extraction
        create_anonymized_pdf=True,  # Create masked PDF
    )
    
    print(f"Ollama available: {reader.ollama_available}")
    print(f"Extracted metadata: {metadata.keys()}")
    print(f"Anonymized PDF: {anonymized_pdf}")


if __name__ == "__main__":
    print("ReportReader PDF Anonymization Examples")
    print("=" * 60)
    print("\nThese examples show different ways to anonymize medical reports:")
    print("1. Text-only anonymization (fast)")
    print("2. PDF with blackened sensitive regions (recommended)")
    print("3. Advanced with region cropping (for analysis)")
    print("4. LLM-enhanced extraction (if Ollama available)")
    print("\n" + "=" * 60)
    
    # Uncomment to run examples:
    # example_basic_anonymization()
    # example_pdf_with_masking()
    # example_with_cropping()
    # example_llm_extraction()
    
    print("\n✅ To use: Uncomment the example functions above and provide real PDF paths")
