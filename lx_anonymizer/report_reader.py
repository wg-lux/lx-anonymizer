from pyexpat import model
from django.core.files import images
from .settings import DEFAULT_SETTINGS
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
from .spacy_extractor import PatientDataExtractor, ExaminerDataExtractor, EndoscopeDataExtractor, ExaminationDataExtractor
from .spacy_regex import PatientDataExtractorLg
from .text_anonymizer import anonymize_text
from .custom_logger import logger
from .name_fallback import extract_patient_info_from_text
from .ocr import tesseract_full_image_ocr, trocr_full_image_ocr_on_boxes # Import OCR fallback
from .pdf_operations import convert_pdf_to_images
from .model_service import model_service
from .donut_ocr import donut_full_image_ocr
from .ensemble_ocr import ensemble_ocr  # Import the new ensemble OCR

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
        self.patient_extractor = PatientDataExtractor()
        self.verbose_patient_extractor = PatientDataExtractorLg()
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
        """
        report_meta = {}
        
        logger.debug(f"Full text extracted from PDF: {text[:500]}...")
        lines = text.split('\n')
        patient_lines = [line for line in lines if self.flags["patient_info_line"] in line]
        logger.debug(f"Found {len(patient_lines)} lines containing patient_info_line: {patient_lines}")
        
        patient_info = None
        for line in lines:
            if self.flags["patient_info_line"] in line:
                logger.debug(f"Processing patient line: {line}")
                patient_info = self.verbose_patient_extractor.extract_patient_info(line)
                logger.debug(f"Verbose extractor result: {patient_info}")
                if not patient_info or patient_info.get('patient_first_name') in [None, "NOT FOUND", "Dr", "Dr."]:
                    logger.debug("Falling back to standard extractor")
                    patient_info = self.patient_extractor.extract_patient_info(line)
                    logger.debug(f"Standard extractor result: {patient_info}")
                if not patient_info or patient_info.get('patient_first_name') in [None, "NOT FOUND", "Dr", "Dr."]:
                    from .name_extractor_utils import extract_name_from_patient_line
                    logger.debug("Trying direct regex extraction")
                    first_name, last_name, birthdate, _ = extract_name_from_patient_line(line)
                    if first_name not in ["Unknown", "Dr", "Dr."] or last_name not in ["Unknown"]:
                        from .determine_gender import determine_gender
                        gender = determine_gender(first_name)
                        patient_info = {
                            "patient_first_name": first_name,
                            "patient_last_name": last_name,
                            "patient_dob": birthdate,
                            "patient_gender": gender
                        }
                        logger.debug(f"Regex extraction result: {patient_info}")
                if patient_info:
                    # Setze konsistente Default-Werte
                    if not patient_info.get('patient_first_name') or patient_info.get('patient_first_name') in ["NOT FOUND", "Dr", "Dr."]:
                        patient_info['patient_first_name'] = "Unknown"
                    if not patient_info.get('patient_last_name') or patient_info.get('patient_last_name') in ["NOT FOUND"]:
                        patient_info['patient_last_name'] = "Unknown"
                    if not patient_info.get('patient_gender') or patient_info.get('patient_gender') in ["NOT FOUND", None]:
                        patient_info['patient_gender'] = "Unknown"
                    report_meta.update(patient_info)
                break
        
        if not patient_info or all(patient_info.get(k) in [None, "Unknown"] for k in ['patient_first_name', 'patient_last_name']):
            logger.debug("Standard extractors failed, using fallback extraction")
            fallback_info = extract_patient_info_from_text(text)
            # Sichert konsistente Werte
            fallback_info.setdefault("patient_first_name", "Unknown")
            fallback_info.setdefault("patient_last_name", "Unknown")
            fallback_info.setdefault("patient_gender", "Unknown")
            report_meta.update(fallback_info)
        
        # Extract examiner information
        for line in lines:
            if self.flags["examiner_info_line"] in line:
                examiner_info = self.examiner_extractor.extract_examiner_info(line)
                if examiner_info:
                    report_meta.update(examiner_info)
                break
        
        # Extract examination information
        for line in lines:
            if "Unters.:" in line or "Eingang am:" in line:
                examination_info = self.examination_extractor.extract_examination_info(line)
                if examination_info:
                    report_meta.update(examination_info)
                break
        
        # Extract endoscope information
        for line in lines:
            if self.flags["endoscope_info_line"] in line:
                endoscope_info = self.endoscope_extractor.extract_endoscope_info(line)
                if endoscope_info:
                    report_meta.update(endoscope_info)
                break

        # Add PDF hash
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
                        # Use the ensemble OCR approach
                        logger.info("Using ensemble OCR approach")
                        ocr_part = ensemble_ocr(pil_image)
                    else:
                        # Default: try multiple OCR methods and choose the best
                        if hasattr(pil_image, "convert"):
                            text_part, _ = tesseract_full_image_ocr(pil_image)
                            text_part_2 = trocr_full_image_ocr_on_boxes(pil_image)
                            text_part_3 = donut_full_image_ocr(pil_image)
                        else:
                            text_part, _ = tesseract_full_image_ocr(str(pil_image))
                            text_part_2 = trocr_full_image_ocr_on_boxes(str(pil_image))
                            text_part_3 = donut_full_image_ocr(str(pil_image))
                        
                        # Choose the OCR result with the most content
                        if len(text_part) >= len(text_part_2) and len(text_part) >= len(text_part_3):
                            ocr_part = text_part
                            logger.info("Selected Tesseract OCR result")
                        elif len(text_part_2) >= len(text_part) and len(text_part_2) >= len(text_part_3):
                            ocr_part = text_part_2
                            logger.info("Selected TrOCR result")
                        else:
                            ocr_part = text_part_3
                            logger.info("Selected GOT OCR result")
                    
                    ocr_text += " " + ocr_part
                
                text = ocr_text.strip()
                logger.info(f"OCR text: {text[:200]}...")
                
                # Apply correction using Ollama with DeepSeek
                if text.strip():  # Only correct if text was recognized
                    logger.info("Applying LLM correction to OCR text via Ollama")
                    try:
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
