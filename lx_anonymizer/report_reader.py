from email.mime import image
import hashlib
import json
import logging
import os
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dateparser
import gender_guesser.detector as gender_detector
import pdfplumber
from faker import Faker
from PIL import Image

from .custom_logger import logger
from .name_fallback import extract_patient_info_from_text
from .ocr import \
    tesseract_full_image_ocr  # , trocr_full_image_ocr_on_boxes # Import OCR fallback
from .ocr_ensemble import ensemble_ocr  # Import the new ensemble OCR
from .ollama_llm_meta_extraction import OllamaOptimizedExtractor
from .pdf_operations import convert_pdf_to_images
from .sensitive_region_cropper import \
    SensitiveRegionCropper  # Import the new cropper
from .settings import DEFAULT_SETTINGS
from .spacy_extractor import (EndoscopeDataExtractor, ExaminationDataExtractor,
                              ExaminerDataExtractor, PatientDataExtractor)
from .text_anonymizer import anonymize_text
from .utils.ollama import ensure_ollama
from .anonymizer import Anonymizer
from .sensitive_meta_interface import SensitiveMeta

class ReportReader:
    def __init__(
        self,
        report_root_path: Optional[str] = None,  # Changed here
        locale: str = DEFAULT_SETTINGS["locale"],  # Changed here, assuming DEFAULT_SETTINGS[\"locale\"] is str
        employee_first_names: Optional[List[str]] = None,  # Changed here
        employee_last_names: Optional[List[str]] = None,  # Changed here
        flags: Optional[Dict[Any, Any]] = None,  # Changed here
        text_date_format: str = DEFAULT_SETTINGS["text_date_format"],  # Changed here, assuming DEFAULT_SETTINGS[\"text_date_format\"] is str
    ):
        self.report_root_path = report_root_path

        self.locale = locale
        self.text_date_format = text_date_format
        self.employee_first_names = employee_first_names if employee_first_names is not None else DEFAULT_SETTINGS["first_names"]
        self.employee_last_names = employee_last_names if employee_last_names is not None else DEFAULT_SETTINGS["last_names"]
        self.flags = flags if flags is not None else DEFAULT_SETTINGS["flags"]
        self.fake = Faker(locale=locale)
        self.gender_detector = gender_detector.Detector(case_sensitive=True)

        # Initialize extractors
        self.patient_extractor = PatientDataExtractor()  # Instantiates the improved class
        self.examiner_extractor = ExaminerDataExtractor()
        self.endoscope_extractor = EndoscopeDataExtractor()
        self.examination_extractor = ExaminationDataExtractor()

        # Initialize sensitive region cropper
        self.sensitive_cropper = SensitiveRegionCropper()
        
        # Initialize Anonymizer
        
        self.anonymizer = Anonymizer()

        # Initialize Ollama (with graceful degradation)
        self.ollama_proc = None
        self.ollama_extractor = None
        self.ollama_available = False
        
        # initialize global sensitive meta

        self.sensitive_meta = SensitiveMeta()
        self.sensitive_meta_dict = self.sensitive_meta.to_dict()

        try:
            self.ollama_proc = ensure_ollama()

            # Try to initialize OllamaOptimizedExtractor
            self.ollama_extractor = OllamaOptimizedExtractor()

            # Check if models are available
            if self.ollama_extractor and self.ollama_extractor.current_model:
                self.ollama_available = True
                logger.info("Ollama LLM features enabled for ReportReader")
            else:
                logger.warning("Ollama models not available, LLM extraction disabled for ReportReader")
                self.ollama_available = False
                self.ollama_extractor = None

        except Exception as e:
            logger.warning(f"Ollama/LLM unavailable for ReportReader, will use SpaCy/regex fallback: {e}")
            self.ollama_available = False
            self.ollama_proc = None
            self.ollama_extractor = None

    def read_pdf(self, pdf_path):
        """
        Read pdf file using pdfplumber and return the raw text content.
        With improved preprocessing for better extraction.
        """
        if pdf_path is None:
            logger.error("PDF path is None, cannot read PDF")
            return ""

        try:
            pdf_path = str(pdf_path)
        except Exception:
            logger.error(f"Cannot convert pdf_path to string: {pdf_path}")
            return ""


        # Disable verbose pdfminer logging
        logging.getLogger("pdfminer").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.psparser").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfdocument").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfinterp").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfpage").setLevel(logging.WARNING)

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                # get the text content of the pdf file
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                # Normalize whitespace
                text = " ".join(text.split())

                # Look for common patient info patterns and enhance visibility for extractors

                # Common patient pattern: enhance formatting
                patient_pattern = r"(Patient|Pat|Patientin|Pat\.)\s*:\s*([A-Za-zäöüÄÖÜß\-]+)\s*[,]\s*([A-Za-zäöüÄÖÜß\-]+)"
                if re.search(patient_pattern, text):
                    text = re.sub(patient_pattern, r"Patient: \2, \3", text)

                # Ensure consistent formatting of birth dates
                dob_pattern = r"(geb|geboren am|Geb\.Dat\.)\s*:?\s*(\d{1,2}\.\d{1,2}\.\d{4})"
                if re.search(dob_pattern, text):
                    text = re.sub(dob_pattern, r"geb. \2", text)

                # Ensure consistent case number formatting
                case_pattern = r"(Fallnummer|Fallnr|Fall\.Nr|Fall-Nr)\s*:?\s*(\d+)"
                if re.search(case_pattern, text):
                    text = re.sub(case_pattern, r"Fallnummer: \2", text)

                logger.debug(f"Enhanced text: {text[:200]}...")

                if not text:
                    warnings.warn(f"Could not read text from {pdf_path}.")

                return text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def extract_report_meta(self, text, pdf_path):
        """
        Extract metadata from report text using the spacy extractor classes.
        Uses the improved PatientDataExtractor and falls back to regex if needed.
        Handles None values correctly and ensures DOB is a datetime object or None.
        """
        report_meta = {}

        logger.debug(f"Full text extracted from PDF: {text[:500]}...")
        lines = text.split("\n") if text else []  # Handle empty text

        patient_info = None
        # Option 1: Try on whole text with Deepseek

        if text:
            patient_info = self.extract_report_meta_deepseek(text)
            if patient_info == {}:
                patient_info = self.patient_extractor(text)  # Use __call__
                logger.debug(f"Patient extractor result on full text: {patient_info}")
        else:
            logger.debug("Skipping extraction on empty text.")
    

        # Check if the result is valid (name found, not None)
        is_valid_info = patient_info and (
            patient_info.get("patient_first_name") is not None or patient_info.get("patient_last_name") is not None
        )  # Check for None

        # Option 2: If full text failed, try line by line with SpaCy
        if not is_valid_info and lines:  # Only run if SpaCy failed AND there are lines
            logger.debug("Extractor failed on full text, trying line by line.")
            for line in lines:
                if re.search(r"pat(?:ient|ientin|\.|iont|bien)", line, re.IGNORECASE):  # Include OCR variants in check
                    logger.debug(f"Processing potential patient line: {line}")
                    patient_info_line = self.patient_extractor(line)
                    logger.debug(f"Patient extractor result on line: {patient_info_line}")
                    # Check if this line gave a valid result (name found, not None)
                    is_valid_line_info = patient_info_line and (
                        patient_info_line.get("patient_first_name") is not None or patient_info_line.get("patient_last_name") is not None
                    )  # Check for None
                    if is_valid_line_info:
                        patient_info = patient_info_line
                        is_valid_info = True
                        break

        if not is_valid_info and text:  # Only run if SpaCy failed AND there is text
            logger.debug("SpaCy extractor failed, using regex fallback extraction")
            fallback_info = extract_patient_info_from_text(text)  # This fallback might still fail on dotless date
            # Use fallback only if it found something SpaCy missed (check against "Unknown")
            if fallback_info.get("patient_first_name") != "Unknown" or fallback_info.get("patient_last_name") != "Unknown":
                logger.debug(f"Regex fallback result: {fallback_info}")
                # Map fallback's "Unknown" to None for consistency before merging
                fallback_info["patient_first_name"] = None if fallback_info.get("patient_first_name") == "Unknown" else fallback_info.get("patient_first_name")
                fallback_info["patient_last_name"] = None if fallback_info.get("patient_last_name") == "Unknown" else fallback_info.get("patient_last_name")
                fallback_info["patient_gender_name"] = None if fallback_info.get("patient_gender_name") == "Unknown" else fallback_info.get("patient_gender_name")
                # Assume fallback might return date as string or None
                patient_info = fallback_info
                is_valid_info = True
            else:
                logger.debug("Regex fallback also found nothing significant.")
                # Ensure patient_info is the blank dict if it was None before fallback
                if not patient_info:
                    patient_info = PatientDataExtractor._blank()

        # Ensure patient_info is initialized if all methods failed or text was empty
        if not patient_info:
            patient_info = PatientDataExtractor._blank()

        # Update report_meta with the final patient_info, ensuring correct DOB type
        # (This part remains largely the same, processing the 'patient_info' dict)
        dob_value = patient_info.get("patient_dob")
        parsed_dob = None
        if isinstance(dob_value, str):
            # Use dateparser for robust parsing, prefer DD.MM.YYYY
            # Specify German date order preference
            parsed_dob = dateparser.parse(dob_value, languages=["de"], settings={"DATE_ORDER": "DMY"})
            if parsed_dob:
                logger.debug(f"Successfully parsed DOB string '{dob_value}' to datetime object: {parsed_dob.date()}")  # Log only date part
                parsed_dob = parsed_dob.date()  # Store only the date part
            else:
                logger.warning(f"Could not parse DOB string '{dob_value}' using dateparser. Setting DOB to None.")
                parsed_dob = None
        elif isinstance(dob_value, datetime):  # Handle if it's already datetime
            parsed_dob = dob_value.date()  # Store only the date part
        elif isinstance(dob_value, date):  # Handle if it's already date
            parsed_dob = dob_value  # Keep as date
        elif dob_value is None:
            parsed_dob = None
        else:
            logger.warning(f"Unexpected type for DOB '{dob_value}' ({type(dob_value)}). Setting DOB to None.")
            parsed_dob = None

        # Ensure final_patient_info uses the parsed_dob (which is now date or None)
        final_patient_info = {
            "patient_first_name": patient_info.get("patient_first_name"),
            "patient_last_name": patient_info.get("patient_last_name"),
            "patient_dob": parsed_dob,  # Use the parsed date object or None
            "casenumber": patient_info.get("casenumber"),
            "patient_gender_name": patient_info.get("patient_gender_name"),
        }
        self.sensitive_meta.safe_update(final_patient_info)
        report_meta = self.sensitive_meta_dict

        # --- Extract other information (Examiner, Examination, Endoscope) ---
        # This part remains the same, using SpaCy extractors on lines
        examiner_found = False
        if lines:  # Only run if lines exist
            for line in lines:
                # Use regex to quickly check if line might contain examiner info
                if re.search(r"unters\w*\s*arzt", line, re.IGNORECASE):
                    examiner_info = self.examiner_extractor.extract_examiner_info(line)
                    if examiner_info:
                        report_meta.update(examiner_info)
                        examiner_found = True

        examination_found = False
        if lines:  # Only run if lines exist
            for line in lines:
                # Use regex to quickly check if line might contain examination date/time info
                if re.search(r"unters\.:|u-datum:|eingang\s*am:", line, re.IGNORECASE):
                    examination_info = self.examination_extractor.extract_examination_info(line)
                    if examination_info:
                        # Ensure examination_date exists before updating
                        if examination_info.get("examination_date"):
                            report_meta.update(examination_info)
                            examination_found = True
                            break  # Assuming only one examination line needed

        endoscope_found = False
        if lines:  # Only run if lines exist
            for line in lines:
                # Use case-insensitive check for the flag
                if self.flags.get("endoscope_info_line", "").lower() in line.lower():
                    endoscope_info = self.endoscope_extractor.extract_endoscope_info(line)
                    if endoscope_info:
                        report_meta.update(endoscope_info)
                        endoscope_found = True
                        break
        
        self.sensitive_meta.safe_update(report_meta)
        report_meta = self.sensitive_meta_dict

        # Add PDF hash (remains the same)
        try:
            if pdf_path and isinstance(pdf_path, (str, os.PathLike)) and os.path.exists(pdf_path):
                with open(str(pdf_path), "rb") as f:
                    pdf_bytes = f.read()
                    pdf_hash_value = self.pdf_hash(pdf_bytes)
                    report_meta["pdf_hash"] = pdf_hash_value
            else:
                logger.warning(f"Cannot calculate PDF hash: invalid or missing pdf_path: {pdf_path}")
                report_meta["pdf_hash"] = None
        except Exception as e:
            logger.error(f"Could not calculate PDF hash for {pdf_path}: {e}")
            report_meta["pdf_hash"] = None

        return report_meta

    def extract_report_meta_deepseek(self, text):
        """Extract metadata using DeepSeek via Ollama structured output."""
        if not self.ollama_available or not self.ollama_extractor:
            logger.warning("Ollama not available for DeepSeek extraction, returning empty dict.")
            return {}

        logger.info("Attempting metadata extraction with DeepSeek (Ollama Structured Output)")
        try:
            # Use the unified extractor that handles retries and fallbacks automatically
            meta_obj = self.ollama_extractor.extract_metadata(text)  # type: ignore
            meta = self.sensitive_meta.safe_update(meta_obj) if meta_obj else None
            if not meta:
                logger.warning("DeepSeek Ollama extraction failed, returning empty dict.")
            else:
                logger.info("DeepSeek Ollama extraction successful.")
            return meta
        except Exception as e:
            logger.warning(f"DeepSeek Ollama extraction error: {e}")
            return None

    def extract_report_meta_medllama(self, text):
        """Extract metadata using MedLLaMA via Ollama structured output."""
        if not self.ollama_available or not self.ollama_extractor:
            logger.warning("Ollama not available for MedLLaMA extraction, returning empty dict.")
            return text

        logger.info("Attempting metadata extraction with MedLLaMA (Ollama Structured Output)")
        try:
            # Use the unified extractor that handles retries and fallbacks automatically
            meta_obj = self.ollama_extractor.extract_metadata(text)  # type: ignore
            meta = self.sensitive_meta.safe_update(meta_obj) if meta_obj else None
            if not meta:
                logger.warning("MedLLaMA Ollama extraction failed, returning empty dict.")
            else:
                logger.info("MedLLaMA Ollama extraction successful.")
            return meta
        except Exception as e:
            logger.warning(f"MedLLaMA Ollama extraction error: {e}")
            return text

    def extract_report_meta_llama3(self, text):
        """Extract metadata using Llama3 via Ollama structured output."""
        if not self.ollama_available or not self.ollama_extractor:
            logger.warning("Ollama not available for Llama3 extraction, returning empty dict.")
            return text

        logger.info("Attempting metadata extraction with Llama3 (Ollama Structured Output)")
        try:
            # Use the unified extractor that handles retries and fallbacks automatically
            meta_obj = self.ollama_extractor.extract_metadata(text)  # type: ignore
            meta = self.sensitive_meta.safe_update(meta_obj) if meta_obj else None
            if not meta:
                logger.warning("Llama3 Ollama extraction failed, returning empty dict.")
            else:
                logger.info("Llama3 Ollama extraction successful.")
            return meta
        except Exception as e:
            logger.warning(f"Llama3 Ollama extraction error: {e}")
            return None

    def anonymize_report(self, text, report_meta):
        """
        Anonymize the report text using the extracted metadata.
        """
        anonymized_text = anonymize_text(
            text=text,
            report_meta=report_meta,
            text_date_format=self.text_date_format,
            lower_cut_off_flags=self.flags["cut_off_below"],
            upper_cut_off_flags=self.flags["cut_off_above"],
            locale=self.locale,
            first_names=self.employee_first_names,
            last_names=self.employee_last_names,
            apply_cutoffs=True,  # Für ReportReader aktivieren wir die Briefkopf-Entfernung
        )

        return anonymized_text



    def process_report(
        self,
        pdf_path=None,
        image_path=None,
        use_ensemble=False,
        verbose=True,
        use_llm_extractor="deepseek",
        text=None,
        create_anonymized_pdf=False,
        anonymized_pdf_output_path=None,
    ):
        """
        Process a report by extracting text, metadata, and creating an anonymized version.
        If the normal pdfplumber extraction fails (or returns very little text), fallback to OCR.
        Optionally, use an ensemble OCR method to improve output quality.
        Optionally, specify an LLM extractor ('deepseek', 'medllama', 'llama3') to use INSTEAD of SpaCy/regex.

        Args:
            pdf_path: Path to the PDF file
            image_path: Path to an image file (alternative to PDF)
            use_ensemble: Whether to use ensemble OCR for better quality
            verbose: Enable verbose logging
            use_llm_extractor: LLM model to use ('deepseek', 'medllama', 'llama3', or None for SpaCy/regex)
            text: Pre-extracted text (optional, will extract from PDF/image if not provided)
            create_anonymized_pdf: If True, creates a PDF with sensitive regions blackened out
            anonymized_pdf_output_path: Custom path for the anonymized PDF (auto-generated if not provided)

        Returns:
            Tuple: (original_text, anonymized_text, report_meta, anonymized_pdf_path)
                - original_text: Extracted text from the document
                - anonymized_text: Text with sensitive information replaced
                - report_meta: Dictionary with extracted metadata (names, dates, etc.)
                - anonymized_pdf_path: Path to the blackened PDF (None if not created)
        """
        
        
        if text is None:
            if not pdf_path and not image_path:
                raise ValueError("Either 'pdf_path' 'image_path' or 'text' must be provided.")
            if isinstance(pdf_path, str):
                pdf_path = Path(pdf_path)
                
            if pdf_path:
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    return "", "", {}
                text = self.read_pdf(pdf_path)
            elif image_path:
                if not isinstance(image_path, (str, os.PathLike)):
                    logger.error(f"Image path must be a string or PathLike object, got {type(image_path)}: {image_path}")
                    return "", "", {}
                if not os.path.exists(image_path):
                    logger.error(f"Image file not found: {image_path}")
                    return "", "", {}, image_path
                # If image_path is provided, we assume it's a single image file
                logger.info(f"Reading text from image file: {image_path}")
                try:
                    pil_image = Image.open(image_path)
                    text, _ = tesseract_full_image_ocr(pil_image)  # Use Tesseract OCR on the image
                except Exception as e:
                    logger.error(f"Error reading image {image_path}: {e}")
                    return "", "", {}, image_path

        ocr_applied = False  # Flag to track if OCR was used

        # --- OCR Fallback ---
        if not text or len(text.strip()) < 50:  # Trigger OCR if text is empty or very short
            ocr_applied = True
            try:
                assert isinstance(text, str)
                logger.info(f"Short/No text detected by pdfplumber ({len(text.strip())} chars), applying OCR fallback.")

                if pdf_path:
                    if not isinstance(pdf_path, (str, os.PathLike, Path)):
                        logger.error(f"Cannot apply OCR: PDF path is not valid: {pdf_path}")
                        return "", "", {}, pdf_path

                    logger.info(f"Converting PDF to images for OCR: {pdf_path}")
                    try:
                        images_from_pdf = convert_pdf_to_images(Path(pdf_path))
                    except Exception as e:
                        logger.error(f"Failed to convert PDF to images: {e}")
                        return "", "", {}, pdf_path
                elif image_path:
                    if not isinstance(image_path, (str, os.PathLike, Path)):
                        logger.error(f"Cannot apply OCR: Image path is not valid: {image_path}")
                        return "", "", {}, image_path
                    try:
                        images_from_pdf = [Image.open(str(image_path))]
                    except Exception as e:
                        logger.error(f"Failed to open image file: {e}")
                        return "", "", {}, image_path
                else:
                    logger.error("No valid path provided for OCR processing")
                    return "", "", {}, pdf_path

                ocr_text = ""

                for idx, pil_image in enumerate(images_from_pdf):
                    # FIX: Validate pil_image before processing
                    if pil_image is None:
                        logger.error(f"Page {idx + 1} image is None, skipping OCR")
                        continue

                    logger.info(f"Processing page {idx + 1} with OCR...")
                    ocr_part = ""
                    if use_ensemble:
                        logger.info("Using ensemble OCR approach")
                        try:
                            ocr_part = ensemble_ocr(pil_image)
                        except Exception as e:
                            logger.error(f"Ensemble OCR failed on page {idx + 1}: {e}")
                            # Fallback to Tesseract if ensemble fails
                            try:
                                logger.info("Falling back to Tesseract OCR after ensemble failure.")
                                text_part, _ = tesseract_full_image_ocr(pil_image)
                                ocr_part = text_part
                            except Exception as te:
                                logger.error(f"Tesseract fallback also failed on page {idx + 1}: {te}")
                                ocr_part = ""
                    else:
                        try:
                            text_part, _ = tesseract_full_image_ocr(pil_image)
                            ocr_part = text_part
                            logger.info(f"Tesseract OCR successful for page {idx + 1}")
                        except Exception as e:
                            logger.error(f"Tesseract OCR failed on page {idx + 1}: {e}")
                            ocr_part = ""

                    ocr_text += " " + ocr_part if ocr_part else ""  # Append only if OCR succeeded

                text = ocr_text.strip()
                logger.info(f"OCR fallback finished. Total text length: {len(text)}. Preview: {text[:200]}...")

                # Apply correction using Ollama (Keep this if you want to correct OCR text)
                if text and len(text.strip()) > 10:  # Only correct if OCR produced something meaningful
                    logger.info("Applying LLM correction to OCR text via Ollama")
                    try:
                        # Use the existing ollama_service for correction
                        from .ollama_service import \
                            ollama_service  # Import locally if needed

                        # Ensure the desired correction model is set up if different from extraction
                        # ollama_service.setup_ollama("deepseek-r1:1.5b") # Or another model
                        corrected_text = ollama_service.correct_ocr_text_in_chunks(text)  # Use chunking

                        if corrected_text and corrected_text != text and len(corrected_text) > 0.5 * len(text):  # Basic sanity check
                            logger.info("OCR text successfully corrected by Ollama.")
                            text = corrected_text
                        elif corrected_text == text:
                            logger.info("Ollama correction resulted in the same text.")
                        else:
                            logger.warning("Ollama OCR correction failed or produced poor result, using original OCR text.")
                    except Exception as e:
                        logger.warning(f"Error using Ollama for correction: {e}")

                if not text or len(text.strip()) < 10:
                    logger.error("OCR fallback produced very short/no text, cannot proceed with metadata extraction.")
                    # Return empty/original text and empty meta if OCR fails badly
                    original_text_from_pdf = self.read_pdf(pdf_path)  # Re-read original for context
                    return original_text_from_pdf, original_text_from_pdf, {}, pdf_path

            except Exception as e:
                logger.error(f"OCR fallback process failed entirely: {e}")
                # Return empty/original text and empty meta if OCR fails badly
                original_text_from_pdf = self.read_pdf(pdf_path)  # Re-read original for context
                return original_text_from_pdf, original_text_from_pdf, {}, pdf_path

        # --- Metadata Extraction ---
        report_meta = {}
        if text and len(text.strip()) >= 10:  # Proceed only if we have some text
            if use_llm_extractor:
                logger.info(f"Using specified LLM extractor: {use_llm_extractor}")
                if use_llm_extractor == "deepseek":
                    report_meta = self.extract_report_meta_deepseek(text)
                elif use_llm_extractor == "medllama":
                    report_meta = self.extract_report_meta_medllama(text)
                elif use_llm_extractor == "llama3":
                    report_meta = self.extract_report_meta_llama3(text)
                else:
                    logger.warning(f"Unknown LLM extractor specified: {use_llm_extractor}. Falling back to default.")
                    report_meta = self.extract_report_meta(text, pdf_path=None)  # Default SpaCy/Regex
                self.sensitive_meta.safe_update(report_meta)
                report_meta = self.sensitive_meta_dict
                # If LLM extraction failed (returned {}), fall back to default SpaCy/Regex
                if not report_meta:
                    logger.warning(f"LLM extractor '{use_llm_extractor}' failed. Falling back to default SpaCy/Regex extraction.")
                    report_meta = self.extract_report_meta(text, pdf_path)

            else:
                # Default extraction: SpaCy + Regex fallback
                logger.info("Using default SpaCy/Regex metadata extraction.")
                report_meta = self.extract_report_meta(text, pdf_path)
                self.sensitive_meta.safe_update(report_meta)
                report_meta = self.sensitive_meta_dict
        else:
            logger.warning("Skipping metadata extraction due to insufficient text content.")
            report_meta = {"pdf_hash": self.pdf_hash(open(str(pdf_path), "rb").read()) if os.path.exists(str(pdf_path)) else None}  # Still add hash if possible
        self.sensitive_meta.safe_update(report_meta)
        # --- Anonymization ---
        anonymized_text = self.anonymize_report(text=text, report_meta=report_meta)

        # --- Create Anonymized PDF (if requested) ---
        anonymized_pdf_path = None

            
        if create_anonymized_pdf and pdf_path:
            try:
                logger.info("Creating anonymized PDF with blackened sensitive regions...")
                anonymized_pdf_path = self.anonymizer.create_anonymized_pdf(pdf_path=str(pdf_path), output_path=anonymized_pdf_output_path, report_meta=report_meta)
                if anonymized_pdf_path:
                    report_meta["anonymized_pdf_path"] = anonymized_pdf_path
                    logger.info(f"Anonymized PDF created: {anonymized_pdf_path}")
                else:
                    logger.warning("Failed to create anonymized PDF")
            except Exception as e:
                logger.error(f"Error creating anonymized PDF: {e}")
                report_meta["anonymized_pdf_error"] = str(e)
        try:
            assert isinstance(text, str)
            report_meta["text"]=text
        except AssertionError as e:
            report_meta["text"]="Unknown"        
        try:
            assert isinstance(anonymized_text, str)
            report_meta["anonymized_text"]=anonymized_text
        except AssertionError as e:
            report_meta["anonymized_text"]="Unknown"
            

        

        sensitive_meta = dict(
            file_path=str(pdf_path) if pdf_path else None,
            patient_first_name=report_meta.get("patient_first_name") or report_meta.get("first_name"),
            patient_last_name=report_meta.get("patient_last_name") or report_meta.get("last_name"),
            patient_dob=report_meta.get("patient_dob") or report_meta.get("birth_date"),
            casenumber=report_meta.get("casenumber") or report_meta.get("casenumber"),
            patient_gender_name=report_meta.get("patient_gender_name") or report_meta.get("gender"),
            examination_date=report_meta.get("examination_date"),
            examination_time=report_meta.get("examination_time"),
            examiner_first_name=report_meta.get("examiner_first_name") or report_meta.get("doctor_first_name"),
            examiner_last_name=report_meta.get("examiner_last_name") or report_meta.get("doctor_last_name"),
            center=report_meta.get("center") or report_meta.get("hospital"),
            text=text,
            anonymized_text=anonymized_text
        )
        self.sensitive_meta.safe_update(sensitive_meta)


        return text, anonymized_text, self.sensitive_meta_dict, anonymized_pdf_path

    def pdf_hash(self, pdf_binary):
        """
        Calculates the SHA256 hash of a PDF file.

        Parameters:
        - pdf_binary: bytes
            The binary content of the PDF file.

        Returns:
        - hash: str
            The SHA256 hash of the PDF file.
        """
        return hashlib.sha256(pdf_binary).hexdigest()

    def process_report_with_cropping(
        self,
        pdf_path=None,
        image_path=None,
        use_ensemble=False,
        verbose=True,
        use_llm_extractor="deepseek",
        text=None,
        crop_output_dir=None,
        crop_sensitive_regions=True,
        anonymization_output_dir=None,
    ):
        """
        Erweiterte Version von process_report mit optionalem Cropping sensitiver Regionen.

        Args:
            pdf_path: Pfad zum PDF
            image_path: Pfad zum Bild (alternative zu PDF)
            use_ensemble: Ob Ensemble-OCR verwendet werden soll
            verbose: Ob ausführliche Logs ausgegeben werden sollen
            use_llm_extractor: LLM-Extraktor für Metadaten ('deepseek', 'medllama', 'llama3')
            text: Bereits extrahierter Text (optional)
            crop_output_dir: Ausgabeverzeichnis für gecropte Regionen
            crop_sensitive_regions: Ob sensitive Regionen gecroppt werden sollen

        Returns:
            Tuple: (original_text, anonymized_text, report_meta, cropped_regions_info)
        """
        # Führe die normale Verarbeitung durch
        original_text, anonymized_text, report_meta, _ = self.process_report(
            pdf_path=pdf_path, image_path=image_path, use_ensemble=use_ensemble, verbose=verbose, use_llm_extractor=use_llm_extractor, text=text
        )

        cropped_regions_info = {}
        anonymized_pdf_path = None

        if not crop_output_dir:
            # Setze ein Standard-Ausgabeverzeichnis, falls nicht angegeben
            crop_output_dir = Path(os.getcwd()).parent / "pdfs" / "cropped_regions"

        if not anonymization_output_dir:
            # Setze ein Standard-Ausgabeverzeichnis für anonymisierte PDFs
            anonymization_output_dir = Path(os.getcwd()).parent / "pdfs" / "anonymized"

        # Führe Cropping durch, falls angefordert
        if crop_sensitive_regions and crop_output_dir and pdf_path:
            try:
                logger.info("Beginne Cropping sensitiver Regionen...")
                cropped_regions_info = self.sensitive_cropper.crop_sensitive_regions(pdf_path=pdf_path, output_dir=str(crop_output_dir))
            except Exception as e:
                logger.error(f"Fehler beim initialien Aufruf der Funktion zum Cropping sensitiver Regionen: {e}")
                cropped_regions_info = {}

            # Erstelle automatisch ein anonymisiertes PDF
            if cropped_regions_info:
                out_dir = Path(anonymization_output_dir) if anonymization_output_dir else Path(pdf_path).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                anonymized_pdf_path = out_dir / (Path(pdf_path).stem + "_anonymized.pdf")
                try:
                    self.sensitive_cropper.create_anonymized_pdf_with_crops(
                        pdf_path=pdf_path, crop_output_dir=str(crop_output_dir), anonymized_pdf_path=str(anonymized_pdf_path)
                    )
                    report_meta["anonymized_pdf_path"] = str(anonymized_pdf_path)
                    logger.info(f"Anonymisiertes PDF erstellt: {anonymized_pdf_path}")
                except Exception as pdf_error:
                    logger.warning(f"Konnte anonymisiertes PDF nicht erstellen: {pdf_error}")
                    report_meta["anonymized_pdf_error"] = str(pdf_error)
                    anonymized_pdf_path = None
            try:
                # Füge Cropping-Informationen zu den Metadaten hinzu
                report_meta["cropped_regions"] = cropped_regions_info
                report_meta["cropping_enabled"] = True

                # Berechne Statistiken
                total_crops = sum(len(crops) for crops in cropped_regions_info.values())
                report_meta["total_cropped_regions"] = total_crops

                logger.info(f"Cropping abgeschlossen: {total_crops} Regionen über {len(cropped_regions_info)} Seiten")
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen von Cropping-Informationen zu den Metadaten: {e}")
                report_meta["cropped_regions"] = {}
                report_meta["total_cropped_regions"] = 0
                report_meta["cropping_error"] = str(e)
                report_meta["cropping_enabled"] = False
                cropped_regions_info = {}
                anonymized_pdf_path = None
                raise e  # Reraise to handle in the outer try-except

        else:
            report_meta["cropping_enabled"] = False

        return original_text, anonymized_text, report_meta, cropped_regions_info, anonymized_pdf_path

    def create_visualization_report(self, pdf_path, output_dir, visualize_all_pages=False):
        """
        Erstellt Visualisierungsreport für sensitive Regionen (für Debugging).

        Args:
            pdf_path: Pfad zum PDF
            output_dir: Ausgabeverzeichnis
            visualize_all_pages: Ob alle Seiten visualisiert werden sollen

        Returns:
            Liste der erstellten Visualisierungsdateien
        """
        from pathlib import Path

        from .ocr import tesseract_full_image_ocr
        from .pdf_operations import convert_pdf_to_images

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_name = Path(pdf_path).stem
        images = convert_pdf_to_images(pdf_path)
        visualization_files = []

        pages_to_process = range(len(images)) if visualize_all_pages else [0]  # Nur erste Seite standardmäßig

        for page_num in pages_to_process:
            if page_num >= len(images):
                continue

            image = images[page_num]
            full_text, word_boxes = tesseract_full_image_ocr(image)

            vis_filename = f"{pdf_name}_page_{page_num + 1}_analysis.png"
            vis_path = output_dir / vis_filename

            self.sensitive_cropper.visualize_sensitive_regions(image, word_boxes, str(vis_path))

            visualization_files.append(str(vis_path))
            logger.info(f"Visualisierung erstellt: {vis_filename}")

        return visualization_files
