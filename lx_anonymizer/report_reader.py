from faker import Faker
import gender_guesser.detector as gender_detector
from typing import List
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
from .model_service import model_service
from .ensemble_ocr import ensemble_ocr  # Import the new ensemble OCR
from .ocr_preprocessing import  preprocess_image, optimize_image_for_medical_text
from datetime import datetime # Add import
import dateparser # Add import

class ReportReader:
    def __init__(
            self,
            report_root_path:str = None, # DEPRECEATED
            locale:str = DEFAULT_SETTINGS["locale"],
            employee_first_names:List[str] = None,
            employee_last_names:List[str] = None,
            flags:dict = None,
            text_date_format:str = DEFAULT_SETTINGS["text_date_format"]
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
    
    def read_pdf(self, pdf_path):
        '''
        Read pdf file using pdfplumber and return the raw text content.
        With improved preprocessing for better extraction.
        '''
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
        lines = text.split('\n')

        patient_info = None
        # Option 1: Try on whole text
        patient_info = self.patient_extractor(text) # Use __call__
        logger.debug(f"Patient extractor result on full text: {patient_info}")

        # Check if the result is valid (name found, not None)
        is_valid_info = patient_info and \
                        (patient_info.get('patient_first_name') is not None or \
                         patient_info.get('patient_last_name') is not None) # Check for None

        # Option 2: If full text failed, try line by line
        if not is_valid_info:
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

        # If SpaCy extractor failed, try the regex fallback
        if not is_valid_info:
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
                 if not patient_info: patient_info = PatientDataExtractor._blank()

        # Update report_meta with the final patient_info, ensuring correct DOB type
        if patient_info:
             dob_value = patient_info.get('patient_dob')
             parsed_dob = None
             if isinstance(dob_value, str):
                 # Use dateparser for robust parsing, prefer DD.MM.YYYY
                 parsed_dob = dateparser.parse(dob_value, date_formats=['%d.%m.%Y', '%Y-%m-%d'])
                 if parsed_dob:
                     logger.debug(f"Successfully parsed DOB string '{dob_value}' to datetime object: {parsed_dob}")
                 else:
                     logger.warning(f"Could not parse DOB string '{dob_value}' using dateparser. Setting DOB to None.")
                     parsed_dob = None
             elif isinstance(dob_value, datetime): # Handle if it's already datetime
                 parsed_dob = dob_value # Keep as datetime
             elif dob_value is None:
                 parsed_dob = None
             else:
                 # Attempt conversion if it's a date object already
                 if hasattr(dob_value, 'year') and hasattr(dob_value, 'month') and hasattr(dob_value, 'day'):
                     try:
                         # Convert date to datetime (assuming time is not relevant)
                         parsed_dob = datetime(dob_value.year, dob_value.month, dob_value.day)
                         logger.debug(f"Converted date object {dob_value} to datetime object: {parsed_dob}")
                     except Exception as e:
                          logger.warning(f"Could not convert date-like object {dob_value} ({type(dob_value)}) to datetime: {e}. Setting DOB to None.")
                          parsed_dob = None
                 else:
                     logger.warning(f"Unexpected type for DOB '{dob_value}' ({type(dob_value)}). Setting DOB to None.")
                     parsed_dob = None

             # Ensure final_patient_info uses the parsed_dob
             final_patient_info = {
                 "patient_first_name": patient_info.get('patient_first_name'),
                 "patient_last_name": patient_info.get('patient_last_name'),
                 "patient_dob": parsed_dob, # Use the parsed datetime object or None
                 "casenumber": patient_info.get('casenumber'),
                 "patient_gender": patient_info.get('patient_gender')
             }
             report_meta.update(final_patient_info)
        else:
             # Ensure report_meta has default keys (all None) even if everything failed
             report_meta.update(PatientDataExtractor._blank()) # dob is already None here


        # --- Extract other information (Examiner, Examination, Endoscope) ---
        examiner_found = False
        for line in lines:
            # Use regex to quickly check if line might contain examiner info
            if re.search(r"unters\w*\s*arzt", line, re.IGNORECASE):
                examiner_info = self.examiner_extractor.extract_examiner_info(line)
                if examiner_info:
                    report_meta.update(examiner_info)
                    examiner_found = True
                    break # Assuming only one examiner line needed

        examination_found = False
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
        for line in lines:
            if self.flags["endoscope_info_line"] in line.lower(): # Check flag case-insensitively
                endoscope_info = self.endoscope_extractor.extract_endoscope_info(line)
                if endoscope_info:
                    report_meta.update(endoscope_info)
                    endoscope_found = True
                    break

        # Add PDF hash (remains the same)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            pdf_hash_value = self.pdf_hash(pdf_bytes)
            report_meta["pdf_hash"] = pdf_hash_value

        return report_meta
    
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
        
    def process_report(self, pdf_path, use_ensemble=False, verbose=True):
        """
        Process a report by extracting text, metadata, and creating an anonymized version.
        If the normal pdfplumber extraction fails (or returns very little text), fallback to OCR.
        Optionally, use an ensemble OCR method to improve output quality.
        """
        text = self.read_pdf(pdf_path)
        if not text.strip():
            try:
                logger.info("No text detected by pdfplumber, applying OCR fallback.")
                images_from_pdf = convert_pdf_to_images(pdf_path)
                ocr_text = ""
                
                for pil_image in images_from_pdf:
                    if use_ensemble:
                        logger.info("Using ensemble OCR approach")
                        try:
                            ocr_part = ensemble_ocr(pil_image)
                        except Exception as e:
                            logger.error(f"Ensemble OCR failed with error: {e}")
                            ocr_part = None  # You may set a default or further fallback if needed.
                    else:
                        # Initialize OCR result variables.
                        text_part = ""
                        text_part_2 = ""
                        text_part_3 = ""
                        
                        # Check if the image is a PIL Image (has "convert" method).
                        if hasattr(pil_image, "convert"):
                            try:
                                text_part, _ = tesseract_full_image_ocr(pil_image)
                            except Exception as e:
                                logger.error(f"Tesseract Text detection failed with error: {e}")
                                text_part = ""

                        else:
                            # Fallback when the input isn't a PIL Image: assume it can be converted to string.
                            try:
                                text_part, _ = tesseract_full_image_ocr(str(pil_image))
                            except Exception as e:
                                logger.error(f"Tesseract Text detection failed with error: {e}")
                                text_part = ""
                            try:
                                # Preprocess the image for better OCR results
                                pil_image = preprocess_image(pil_image, methods=['grayscale', 'denoise', 'contrast', 'sharpen'])
                                text_part_2 = tesseract_full_image_ocr(str(pil_image))
                            except Exception as e:
                                logger.error(f"Preprocessing Text detection before ocr failed with error: {e}")
                                text_part_2 = ""
                            try:
                                # Optimize the image for medical text
                                pil_image = optimize_image_for_medical_text(pil_image)
                                text_part_3 = tesseract_full_image_ocr(str(pil_image))
                            except Exception as e:
                                logger.error(f"Optimizing Text detection before ocr failed with error: {e}")
                                text_part_3 = ""
                                


                        # Choose the OCR result with the most content.
                        if len(text_part) >= len(text_part_2) and len(text_part) >= len(text_part_3):
                            ocr_part = text_part
                            logger.info("Selected Tesseract OCR result")
                        elif len(text_part_2) >= len(text_part) and len(text_part_2) >= len(text_part_3):
                            ocr_part = text_part_2
                            logger.info("Selected Tesseract OCR result with default preprocessing")
                        else:
                            ocr_part = text_part_3
                            logger.info("Selected Tesseract OCR result with optimization for medical text")


                    ocr_text += " " + ocr_part
                
                text = ocr_text.strip()
                logger.info(f"OCR text: {text[:200]}...")
                
                # Apply correction using Ollama with DeepSeek
                if text.strip():  # Only correct if text was recognized
                    logger.info("Applying LLM correction to OCR text via Ollama")
                    try:
                        model_service.setup_ollama("deepseek-r1:1.5b")
                        corrected_text = model_service.correct_text_with_ollama(text)
                        # Check if correction was successful (doesn't start with an error message)
                        if corrected_text and not corrected_text.startswith("Error:"):
                            # Additional checks for correction quality
                            if len(corrected_text) < len(text):
                                logger.warning("OCR correction produced shorter text")
                                model_service.setup_ollama("rjmalagon/medllama3-v20:fp16")
                                corrected_text = model_service.correct_text_with_ollama(corrected_text)
                                model_service.cleanup_models()
                                if len(corrected_text) < len(text):
                                    logger.warning("OCR correction produced shorter text again")
                                    corrected_text = text
                                    model_service.setup_ollama("deepseek-r1:1.5b")
                                    corrected_text = model_service.correct_text_with_ollama_in_chunks(corrected_text)
                                    model_service.cleanup_models()
                                    if len(corrected_text) < len(text):
                                        logger.warning("OCR correction produced shorter text after multiple attempts")
                                        corrected_text = text
                            text = corrected_text
                            logger.info("OCR text successfully corrected.")
                        else:
                            logger.warning("OCR correction failed, using original OCR text.")
                    except Exception as e:
                        logger.warning(f"Error using Ollama for correction: {e}")
                
                if len(text) < 10:
                    logger.error("OCR fallback produced very short text, skipping anonymization.")
                    return text, text, {}
            
            except Exception as e:
                logger.error(f"OCR fallback failed: {e}")
        
        report_meta = self.extract_report_meta(text, pdf_path)
        anonymized_text = self.anonymize_report(text=text, report_meta=report_meta)
        
        logger.debug(f"Anonymized text: {anonymized_text}")
        
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
