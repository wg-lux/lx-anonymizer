# This module has a strong focus on anonymizing video and image data
- Main Entrypoint Video: /lx_anonymizer/frame_cleaner.py
- Main Entrypoint for Reports:  /lx_anonymizer/report_reader.py
- Study Data about the LLM usage in the pipeline: ./study-data

## The pipeline contains

Multi-Detector Cascade: EAST, Tesseract, and CRAFT for robust localization.

Fuzzy-Spatial Coupling: text correction and bounding box resizing.

Heuristic Selection: BestFrameText scoring and LLM integration for identifying the most relevant data.

Anonymization Strategy: Blurring + NER.

Standardized Output: /home/admin/lx-anonymizer/lx_anonymizer/sensitive_meta_interface.py

