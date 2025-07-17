from pyexpat import model
from faker import Faker
import gender_guesser.detector as gender_detector
from typing import List, Optional, Dict, Any # Added Optional, Dict, Any
import os
import warnings
import hashlib
import pdfplumber
from PIL import Image
from uuid import uuid4
import json
import re
from .settings import DEFAULT_SETTINGS
from .spacy_extractor import PatientDataExtractor, ExaminerDataExtractor, EndoscopeDataExtractor, ExaminationDataExtractor
from .text_anonymizer import anonymize_text
from .custom_logger import logger
from .name_fallback import extract_patient_info_from_text
from .ocr import tesseract_full_image_ocr #, trocr_full_image_ocr_on_boxes # Import OCR fallback
from .pdf_operations import convert_pdf_to_images
from .ensemble_ocr import ensemble_ocr  # Import the new ensemble OCR
from .ocr_preprocessing import preprocess_image, optimize_image_for_medical_text
from datetime import datetime, date
import dateparser
from .utils.ollama import ensure_ollama
from .ollama_llm_meta_extraction import extract_with_fallback

class ReportReader:
    def __init__(
            self,
            report_root_path: Optional[str] = None, # Changed here
            locale: str = DEFAULT_SETTINGS["locale"], # Changed here, assuming DEFAULT_SETTINGS[\"locale\"] is str
            employee_first_names: Optional[List[str]] = None, # Changed here
            employee_last_names: Optional[List[str]] = None, # Changed here
            flags: Optional[Dict[Any, Any]] = None, # Changed here
            text_date_format: str = DEFAULT_SETTINGS["text_date_format"] # Changed here, assuming DEFAULT_SETTINGS[\"text_date_format\"] is str
    ):
        self.report_root_path = report_root_path

        self.locale = locale
        self.text_date_format = text_date_format
        self.employee_first_names = employee_first_names if employee_first_names is not None else DEFAULT_SETTINGS["first_names"]
        self.employee_last_names = employee_last_names if employee_last_names is not None else DEFAULT_SETTINGS["last_names"]
        self.flags = flags if flags is not None else DEFAULT_SETTINGS["flags"]
        self.fake = Faker(locale=locale)
        self.gender_detector = gender_detector.Detector(case_sensitive = True)
        
        # Initialize extractors
        self.patient_extractor = PatientDataExtractor() # Instantiates the improved class
        self.examiner_extractor = ExaminerDataExtractor()
        self.endoscope_extractor = EndoscopeDataExtractor()
        self.examination_extractor = ExaminationDataExtractor()
        
        # Initialize Ollama
        self.ollama_proc = ensure_ollama()
        
    
    def read_pdf(self, pdf_path):
        '''
        Read pdf file using pdfplumber and return the raw text content.
        With improved preprocessing for better extraction.
        '''
        if pdf_path is None:
            logger.error("PDF path is None, cannot read PDF")
            return ""
        
        if not isinstance(pdf_path, (str, os.PathLike)):
            logger.error(f"PDF path must be a string or PathLike object, got {type(pdf_path)}: {pdf_path}")
            return ""
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
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
        lines = text.split('\n') if text else [] # Handle empty text

        patient_info = None
        # Option 1: Try on whole text with Deepseek
        
        if text:
            patient_info = self.extract_report_meta_deepseek(text)
            if patient_info == {}:
                patient_info = self.patient_extractor(text) # Use __call__
                logger.debug(f"Patient extractor result on full text: {patient_info}")
        else:
            logger.debug("Skipping extraction on empty text.")
            patient_info = PatientDataExtractor # Start with blank if no text

        # Check if the result is valid (name found, not None)
        is_valid_info = patient_info and \
                        (patient_info.get('patient_first_name') is not None or \
                         patient_info.get('patient_last_name') is not None) # Check for None

        # Option 2: If full text failed, try line by line with SpaCy
        if not is_valid_info and lines: # Only run if SpaCy failed AND there are lines
            logger.debug("Extractor failed on full text, trying line by line.")
            for line in lines:
                 if re.search(r"pat(?:ient|ientin|\.|iont|bien)", line, re.IGNORECASE): # Include OCR variants in check
                     logger.debug(f"Processing potential patient line: {line}")
                     patient_info_line = self.patient_extractor(line)
                     logger.debug(f"Patient extractor result on line: {patient_info_line}")
                     # Check if this line gave a valid result (name found, not None)
                     is_valid_line_info = patient_info_line and \
                                          (patient_info_line.get('patient_first_name') is not None or \
                                           patient_info_line.get('patient_last_name') is not None) # Check for None
                     if is_valid_line_info:
                         patient_info = patient_info_line
                         is_valid_info = True
                         break

        if not is_valid_info and text: # Only run if SpaCy failed AND there is text
            logger.debug("SpaCy extractor failed, using regex fallback extraction")
            fallback_info = extract_patient_info_from_text(text) # This fallback might still fail on dotless date
            # Use fallback only if it found something SpaCy missed (check against "Unknown")
            if fallback_info.get("patient_first_name") != "Unknown" or fallback_info.get("patient_last_name") != "Unknown":
                 logger.debug(f"Regex fallback result: {fallback_info}")
                 # Map fallback's "Unknown" to None for consistency before merging
                 fallback_info['patient_first_name'] = None if fallback_info.get('patient_first_name') == "Unknown" else fallback_info.get('patient_first_name')
                 fallback_info['patient_last_name'] = None if fallback_info.get('patient_last_name') == "Unknown" else fallback_info.get('patient_last_name')
                 fallback_info['patient_gender'] = None if fallback_info.get('patient_gender') == "Unknown" else fallback_info.get('patient_gender')
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
        dob_value = patient_info.get('patient_dob')
        parsed_dob = None
        if isinstance(dob_value, str):
            # Use dateparser for robust parsing, prefer DD.MM.YYYY
            # Specify German date order preference
            parsed_dob = dateparser.parse(dob_value, languages=['de'], settings={'DATE_ORDER': 'DMY'})
            if parsed_dob:
                logger.debug(f"Successfully parsed DOB string '{dob_value}' to datetime object: {parsed_dob.date()}") # Log only date part
                parsed_dob = parsed_dob.date() # Store only the date part
            else:
                logger.warning(f"Could not parse DOB string '{dob_value}' using dateparser. Setting DOB to None.")
                parsed_dob = None
        elif isinstance(dob_value, datetime): # Handle if it's already datetime
            parsed_dob = dob_value.date() # Store only the date part
        elif isinstance(dob_value, date): # Handle if it's already date
             parsed_dob = dob_value # Keep as date
        elif dob_value is None:
            parsed_dob = None
        else:
            logger.warning(f"Unexpected type for DOB '{dob_value}' ({type(dob_value)}). Setting DOB to None.")
            parsed_dob = None

        # Ensure final_patient_info uses the parsed_dob (which is now date or None)
        final_patient_info = {
            "patient_first_name": patient_info.get('patient_first_name'),
            "patient_last_name": patient_info.get('patient_last_name'),
            "patient_dob": parsed_dob, # Use the parsed date object or None
            "casenumber": patient_info.get('casenumber'),
            "patient_gender": patient_info.get('patient_gender')
        }
        report_meta.update(final_patient_info)


        # --- Extract other information (Examiner, Examination, Endoscope) ---
        # This part remains the same, using SpaCy extractors on lines
        examiner_found = False
        if lines: # Only run if lines exist
            for line in lines:
                # Use regex to quickly check if line might contain examiner info
                if re.search(r"unters\w*\s*arzt", line, re.IGNORECASE):
                    examiner_info = self.examiner_extractor.extract_examiner_info(line)
                    if examiner_info:
                        report_meta.update(examiner_info)
                        examiner_found = True
                        break # Assuming only one examiner line needed

        examination_found = False
        if lines: # Only run if lines exist
            for line in lines:
                 # Use regex to quickly check if line might contain examination date/time info
                 if re.search(r"unters\.:|u-datum:|eingang\s*am:", line, re.IGNORECASE):
                    examination_info = self.examination_extractor.extract_examination_info(line)
                    if examination_info:
                        # Ensure examination_date exists before updating
                        if examination_info.get('examination_date'):
                             report_meta.update(examination_info)
                             examination_found = True
                             break # Assuming only one examination line needed

        endoscope_found = False
        if lines: # Only run if lines exist
            for line in lines:
                # Use case-insensitive check for the flag
                if self.flags.get("endoscope_info_line", "").lower() in line.lower():
                    endoscope_info = self.endoscope_extractor.extract_endoscope_info(line)
                    if endoscope_info:
                        report_meta.update(endoscope_info)
                        endoscope_found = True
                        break

        # Add PDF hash (remains the same)
        try:
            # FIX: Validate pdf_path before calculating hash
            if pdf_path and isinstance(pdf_path, (str, os.PathLike)) and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
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
        logger.info("Attempting metadata extraction with DeepSeek (Ollama Structured Output)")
        # Use the wrapper that handles retries and returns {} on failure
        meta = extract_with_fallback(text, model="deepseek-r1:1.5b")
        if not meta:
            logger.warning("DeepSeek Ollama extraction failed, returning empty dict.")
        else:
             logger.info("DeepSeek Ollama extraction successful.")
        return meta

    def extract_report_meta_medllama(self, text):
        """Extract metadata using MedLLaMA via Ollama structured output."""
        logger.info("Attempting metadata extraction with MedLLaMA (Ollama Structured Output)")
        # Use the wrapper that handles retries and returns {} on failure
        meta = extract_with_fallback(text, model="rjmalagon/medllama3-v20:fp16")
        if not meta:
            logger.warning("MedLLaMA Ollama extraction failed, returning empty dict.")
        else:
             logger.info("MedLLaMA Ollama extraction successful.")
        return meta

    def extract_report_meta_llama3(self, text):
        """Extract metadata using Llama3 via Ollama structured output."""
        logger.info("Attempting metadata extraction with Llama3 (Ollama Structured Output)")
        # Use the wrapper that handles retries and returns {} on failure
        meta = extract_with_fallback(text, model="llama3:8b") # Or llama3:70b if available/needed
        if not meta:
            logger.warning("Llama3 Ollama extraction failed, returning empty dict.")
        else:
             logger.info("Llama3 Ollama extraction successful.")
        return meta

    def anonymize_report(self, text, report_meta):
        """
        Anonymize the report text using the extracted metadata.
        """
        anonymized_text = anonymize_text(
            text = text,
            report_meta = report_meta,
            text_date_format = self.text_date_format,
            lower_cut_off_flags = self.flags["cut_off_below"],
            upper_cut_off_flags = self.flags["cut_off_above"],
            locale = self.locale,
            first_names = self.employee_first_names,
            last_names = self.employee_last_names,
            apply_cutoffs = True  # Für ReportReader aktivieren wir die Briefkopf-Entfernung
        )

        return anonymized_text
        
    def process_report(self, pdf_path=None, image_path=None, use_ensemble=False, verbose=True, use_llm_extractor='deepseek', text=None):
        """
        Process a report by extracting text, metadata, and creating an anonymized version.
        If the normal pdfplumber extraction fails (or returns very little text), fallback to OCR.
        Optionally, use an ensemble OCR method to improve output quality.
        Optionally, specify an LLM extractor ('deepseek', 'medllama', 'llama3') to use INSTEAD of SpaCy/regex.
        """
        if text is None:
            if not pdf_path and not image_path:
                raise ValueError("Either 'pdf_path' 'image_path' or 'text' must be provided.")
            
            # FIX: Validate paths before proceeding
            if pdf_path:
                if not isinstance(pdf_path, (str, os.PathLike)):
                    logger.error(f"PDF path must be a string or PathLike object, got {type(pdf_path)}: {pdf_path}")
                    return "", "", {}
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
                    return "", "", {}
                # If image_path is provided, we assume it's a single image file
                logger.info(f"Reading text from image file: {image_path}")
                try:
                    pil_image = Image.open(image_path)
                    text, _ = tesseract_full_image_ocr(pil_image) # Use Tesseract OCR on the image
                except Exception as e:
                    logger.error(f"Error reading image {image_path}: {e}")
                    return "", "", {}

        
        
        ocr_applied = False # Flag to track if OCR was used

        # --- OCR Fallback ---
        if not text or len(text.strip()) < 50: # Trigger OCR if text is empty or very short
            ocr_applied = True
            try:
                logger.info(f"Short/No text detected by pdfplumber ({len(text.strip())} chars), applying OCR fallback.")
                
                # FIX: Validate pdf_path before OCR processing
                if pdf_path:
                    if not isinstance(pdf_path, (str, os.PathLike)):
                        logger.error(f"Cannot apply OCR: PDF path is not valid: {pdf_path}")
                        return "", "", {}
                    if not os.path.exists(pdf_path):
                        logger.error(f"Cannot apply OCR: PDF file not found: {pdf_path}")
                        return "", "", {}
                    
                    logger.info(f"Converting PDF to images for OCR: {pdf_path}")
                    try:
                        images_from_pdf = convert_pdf_to_images(pdf_path)
                    except Exception as e:
                        logger.error(f"Failed to convert PDF to images: {e}")
                        return "", "", {}
                elif image_path:
                    if not isinstance(image_path, (str, os.PathLike)):
                        logger.error(f"Cannot apply OCR: Image path is not valid: {image_path}")
                        return "", "", {}
                    if not os.path.exists(image_path):
                        logger.error(f"Cannot apply OCR: Image file not found: {image_path}")
                        return "", "", {}
                    try:
                        images_from_pdf = [Image.open(image_path)]
                    except Exception as e:
                        logger.error(f"Failed to open image file: {e}")
                        return "", "", {}
                else:
                    logger.error("No valid path provided for OCR processing")
                    return "", "", {}
                
                ocr_text = ""

                for idx, pil_image in enumerate(images_from_pdf):
                    # FIX: Validate pil_image before processing
                    if pil_image is None:
                        logger.error(f"Page {idx+1} image is None, skipping OCR")
                        continue
                        
                    logger.info(f"Processing page {idx+1} with OCR...")
                    ocr_part = ""
                    if use_ensemble:
                        logger.info("Using ensemble OCR approach")
                        try:
                            ocr_part = ensemble_ocr(pil_image)
                        except Exception as e:
                            logger.error(f"Ensemble OCR failed on page {idx+1}: {e}")
                            # Fallback to Tesseract if ensemble fails
                            try:
                                logger.info("Falling back to Tesseract OCR after ensemble failure.")
                                text_part, _ = tesseract_full_image_ocr(pil_image)
                                ocr_part = text_part
                            except Exception as te:
                                logger.error(f"Tesseract fallback also failed on page {idx+1}: {te}")
                                ocr_part = ""
                    else:
                        try:
                            text_part, _ = tesseract_full_image_ocr(pil_image)
                            ocr_part = text_part
                            logger.info(f"Tesseract OCR successful for page {idx+1}")
                        except Exception as e:
                            logger.error(f"Tesseract OCR failed on page {idx+1}: {e}")
                            ocr_part = ""

                    ocr_text += " " + ocr_part if ocr_part else "" # Append only if OCR succeeded

                text = ocr_text.strip()
                logger.info(f"OCR fallback finished. Total text length: {len(text)}. Preview: {text[:200]}...")

                # Apply correction using Ollama (Keep this if you want to correct OCR text)
                if text and len(text.strip()) > 10: # Only correct if OCR produced something meaningful
                    logger.info("Applying LLM correction to OCR text via Ollama")
                    try:
                        # Use the existing ollama_service for correction
                        from .ollama_service import ollama_service # Import locally if needed
                        # Ensure the desired correction model is set up if different from extraction
                        # ollama_service.setup_ollama("deepseek-r1:1.5b") # Or another model
                        corrected_text = ollama_service.correct_ocr_text_in_chunks(text) # Use chunking

                        if corrected_text and corrected_text != text and len(corrected_text) > 0.5 * len(text): # Basic sanity check
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
                    original_text_from_pdf = self.read_pdf(pdf_path) # Re-read original for context
                    return original_text_from_pdf, original_text_from_pdf, {}

            except Exception as e:
                logger.error(f"OCR fallback process failed entirely: {e}")
                # Return empty/original text and empty meta if OCR fails badly
                original_text_from_pdf = self.read_pdf(pdf_path) # Re-read original for context
                return original_text_from_pdf, original_text_from_pdf, {}

        # --- Metadata Extraction ---
        report_meta = {}
        if text and len(text.strip()) >= 10: # Proceed only if we have some text
            if use_llm_extractor:
                logger.info(f"Using specified LLM extractor: {use_llm_extractor}")
                if use_llm_extractor == 'deepseek':
                    report_meta = self.extract_report_meta_deepseek(text)
                elif use_llm_extractor == 'medllama':
                    report_meta = self.extract_report_meta_medllama(text)
                elif use_llm_extractor == 'llama3':
                    report_meta = self.extract_report_meta_llama3(text)
                else:
                    logger.warning(f"Unknown LLM extractor specified: {use_llm_extractor}. Falling back to default.")
                    report_meta = self.extract_report_meta(text, pdf_path=None) # Default SpaCy/Regex

                # If LLM extraction failed (returned {}), fall back to default SpaCy/Regex
                if not report_meta:
                    logger.warning(f"LLM extractor '{use_llm_extractor}' failed. Falling back to default SpaCy/Regex extraction.")
                    report_meta = self.extract_report_meta(text, pdf_path)

            else:
                # Default extraction: SpaCy + Regex fallback
                logger.info("Using default SpaCy/Regex metadata extraction.")
                report_meta = self.extract_report_meta(text, pdf_path)
        else:
             logger.warning("Skipping metadata extraction due to insufficient text content.")
             report_meta = {"pdf_hash": self.pdf_hash(open(pdf_path, "rb").read()) if os.path.exists(pdf_path) else None} # Still add hash if possible


        # --- Anonymization ---
        anonymized_text = self.anonymize_report(text=text, report_meta=report_meta)

        # Log final outcome
        if ocr_applied:
             logger.info(f"Processed (OCR applied): {pdf_path}. Meta keys found: {list(report_meta.keys())}")
        else:
             logger.info(f"Processed (No OCR): {pdf_path}. Meta keys found: {list(report_meta.keys())}")
        logger.debug(f"Final Report Meta: {report_meta}")
        # logger.debug(f"Anonymized text preview: {anonymized_text[:200]}...") # Be careful logging anonymized text

        return text, anonymized_text, report_meta
    
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
