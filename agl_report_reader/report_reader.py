from .settings import DEFAULT_SETTINGS
from faker import Faker
import gender_guesser.detector as gender_detector
from typing import List
import os
import warnings

import pdfplumber
from uuid import uuid4
import json
import os
from .lx_anonymizer.spacy_extractor import PatientDataExtractor, ExaminerDataExtractor, EndoscopeDataExtractor, ExaminationDataExtractor
from .lx_anonymizer.text_anonymizer import anonymize_text
from .utils import pdf_hash

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
        self.examiner_extractor = ExaminerDataExtractor()
        self.endoscope_extractor = EndoscopeDataExtractor()
        self.examination_extractor = ExaminationDataExtractor()
    
    def read_pdf(self, pdf_path):
        '''
        Read pdf file using pdfplumber and return the raw text content.
        '''
        with pdfplumber.open(pdf_path) as pdf:
            # get the text content of the pdf file
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            
            
            raw_text = text

        if not raw_text:
            warnings.warn(f"Could not read text from {pdf_path}.")
            return text
        
        return text
    
        
    def extract_report_meta(self, text, pdf_path):
        """
        Extract metadata from report text using the spacy extractor classes.
        """
        report_meta = {}
        
        # Split text into lines for processing
        lines = text.split('\n')
        
        # Extract patient information
        for line in lines:
            if self.flags["patient_info_line"] in line:
                patient_info = self.patient_extractor.extract_patient_info(line)
                if patient_info:
                    report_meta.update(patient_info)
                break
        
        # Extract endoscope information
        for line in lines:
            if self.flags["endoscope_info_line"] in line:
                endoscope_info = self.endoscope_extractor.extract_endoscope_info(line)
                if endoscope_info:
                    report_meta.update(endoscope_info)
                break
        
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

        # Add PDF hash
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            pdf_hash_value = pdf_hash(pdf_bytes)
            report_meta["pdf_hash"] = pdf_hash_value

        return report_meta
    
    def anonymize_report(
            self,
            text,
            report_meta
    ):
        anonymized_text = anonymize_report(
            text = text,
            report_meta = report_meta,
            text_date_format = self.text_date_format,
            lower_cut_off_flags=self.flags["cut_off_below"],
            upper_cut_off_flags=self.flags["cut_off_above"],
            locale = self.locale,
            first_names = self.employee_first_names,
            last_names = self.employee_last_names
        )

        return anonymized_text
        

    def process_report(
        self,
        pdf_path,
        verbose = True
    ):
    
        text = self.read_pdf(pdf_path)
        report_meta = self.extract_report_meta(
            text,
            pdf_path
        )
        anonymized_text = self.anonymize_report(
            text = text,
            report_meta = report_meta,
        )


        return text, anonymized_text, report_meta

