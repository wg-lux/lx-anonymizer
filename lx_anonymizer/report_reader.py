from .settings import DEFAULT_SETTINGS
from faker import Faker
import gender_guesser.detector as gender_detector
from typing import List
import os

import pdfplumber
from uuid import uuid4
import json
import os
from .anonymization import anonymize_report
from .extraction import extract_report_meta
import warnings
from .utils import pdf_hash

class ReportReader:
    def __init__(
            self,
            report_root_path:str = None, # DEPRECEATED
            locale:str = DEFAULT_SETTINGS["locale"],
            employee_first_names:List[str] = DEFAULT_SETTINGS["first_names"],
            employee_last_names:List[str] = DEFAULT_SETTINGS["last_names"],
            flags:List[str] = DEFAULT_SETTINGS["flags"],
            text_date_format:str = DEFAULT_SETTINGS["text_date_format"]
    ):
        self.report_root_path = report_root_path

        self.locale = locale
        self.text_date_format = text_date_format
        self.employee_first_names = employee_first_names
        self.employee_last_names = employee_last_names
        self.flags = flags
        self.fake = Faker(locale=locale)
        self.gender_detector = gender_detector.Detector(case_sensitive = True)
    
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
        report_meta = extract_report_meta(
            text,
            patient_info_line_flag = self.flags["patient_info_line"],
            endoscope_info_line_flag = self.flags["endoscope_info_line"],
            examiner_info_line_flag = self.flags["examiner_info_line"],
            gender_detector=self.gender_detector
        )

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
    
