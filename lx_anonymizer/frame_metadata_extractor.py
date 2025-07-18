"""
Frame-specific metadata extraction module for video processing.

This module provides specialized metadata extraction functionality optimized for video frames:
- Direct pattern matching for frame overlays
- Medical terminology recognition
- Patient data extraction from frame text
- Optimized for real-time frame processing

Separated from PDF processing logic to maintain clean architecture.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, date
import dateparser

logger = logging.getLogger(__name__)

class FrameMetadataExtractor:
    """
    Specialized metadata extractor for video frame text processing.
    
    Optimized for:
    - Fast pattern matching on frame overlay text
    - Medical information extraction
    - Patient data detection
    - German medical terminology
    """
    
    def __init__(self):
        # Frame-specific patterns optimized for overlay text
        self.patient_patterns = [
            r'Patient[:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)',
            r'Pat[\.:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)',
            r'Name[:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)',
            r'([A-Za-zäöüÄÖÜß]{2,})\s*,\s*([A-Za-zäöüÄÖÜß]{2,})',  # Last, First format
        ]
        
        self.dob_patterns = [
            r'geb[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'geboren[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'Geb\.Dat[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'DOB[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'(\d{1,2}\.\d{1,2}\.\d{2,4})',  # Standalone date
        ]
        
        self.case_patterns = [
            r'Fall[nr]*[\.:\s]*(\d+)',
            r'Case[:\s]*(\d+)',
            r'Fallnummer[:\s]*(\d+)',
            r'ID[:\s]*(\d+)',
        ]
        
        self.date_patterns = [
            r'Datum[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'Date[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'Untersuchung[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'(\d{1,2}\.\d{1,2}\.\d{2,4})',  # Any date format
        ]
        
        self.time_patterns = [
            r'Zeit[:\s]*(\d{1,2}:\d{2})',
            r'Time[:\s]*(\d{1,2}:\d{2})',
            r'(\d{1,2}:\d{2})',  # Any time format
        ]
        
        self.examiner_patterns = [
            r'Arzt[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]+)',
            r'Dr[\.:\s]*([A-Za-zäöüÄÖÜß\s\-\.]+)',
            r'Untersucher[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]+)',
            r'Examiner[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]+)',
        ]
        
        self.gender_patterns = [
            r'(männlich|weiblich|male|female|m|f|w)',
        ]
    
    def extract_metadata_from_frame_text(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from frame OCR text using specialized patterns.
        
        Args:
            text: OCR-extracted text from frame
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender": None,
            "examination_date": None,
            "examination_time": None,
            "examiner": None,
            "source": "frame_extraction"
        }
        
        
        
        if not text or not text.strip():
            return metadata
        
        
        
        try:
            # Extract patient names
            first_name, last_name = self._extract_patient_names(text)
            if first_name:
                metadata["patient_first_name"] = first_name
            if last_name:
                metadata["patient_last_name"] = last_name
            
            # Extract date of birth
            dob = self._extract_date_of_birth(text)
            if dob:
                metadata["patient_dob"] = dob
            
            # Extract case number
            case_num = self._extract_case_number(text)
            if case_num:
                metadata["casenumber"] = case_num
            
            # Extract examination date
            exam_date = self._extract_examination_date(text)
            if exam_date:
                metadata["examination_date"] = exam_date
            
            # Extract examination time
            exam_time = self._extract_examination_time(text)
            if exam_time:
                metadata["examination_time"] = exam_time
            
            # Extract examiner
            examiner = self._extract_examiner(text)
            if examiner:
                metadata["examiner"] = examiner
            
            # Extract gender
            gender = self._extract_gender(text)
            if gender:
                metadata["patient_gender"] = gender
            
            return metadata
            
        except Exception as e:
            logger.error(f"Frame metadata extraction failed: {e}")
            return metadata
    
    def _extract_patient_names(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient first and last names from text."""
        try:
            for pattern in self.patient_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple) and len(matches[0]) >= 2:
                        # Handle tuple matches (first, last) or (last, first)
                        name1, name2 = matches[0][:2]
                        name1, name2 = name1.strip(), name2.strip()
                        
                        # Determine which is first/last based on pattern
                        if ',' in text and text.index(',') < text.index(name2):
                            # "Last, First" format
                            return name2, name1
                        else:
                            # "First Last" format
                            return name1, name2
                    elif isinstance(matches[0], str):
                        # Single name found
                        return matches[0].strip(), None
            
            return None, None
            
        except Exception as e:
            logger.error(f"Patient name extraction failed: {e}")
            return None, None
    
    def _extract_date_of_birth(self, text: str) -> Optional[date]:
        """Extract date of birth from text."""
        try:
            for pattern in self.dob_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    date_str = matches[0].strip()
                    parsed_date = self._parse_german_date(date_str)
                    if parsed_date:
                        return parsed_date
            
            return None
            
        except Exception as e:
            logger.error(f"DOB extraction failed: {e}")
            return None
    
    def _extract_case_number(self, text: str) -> Optional[str]:
        """Extract case number from text."""
        try:
            for pattern in self.case_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0].strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Case number extraction failed: {e}")
            return None
    
    def _extract_examination_date(self, text: str) -> Optional[date]:
        """Extract examination date from text."""
        try:
            for pattern in self.date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    date_str = matches[0].strip()
                    parsed_date = self._parse_german_date(date_str)
                    if parsed_date:
                        return parsed_date
            
            return None
            
        except Exception as e:
            logger.error(f"Examination date extraction failed: {e}")
            return None
    
    def _extract_examination_time(self, text: str) -> Optional[str]:
        """Extract examination time from text."""
        try:
            for pattern in self.time_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    time_str = matches[0].strip()
                    # Validate time format
                    if re.match(r'^\d{1,2}:\d{2}$', time_str):
                        return time_str
            
            return None
            
        except Exception as e:
            logger.error(f"Examination time extraction failed: {e}")
            return None
    
    def _extract_examiner(self, text: str) -> Optional[str]:
        """Extract examiner name from text."""
        try:
            for pattern in self.examiner_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    examiner = matches[0].strip()
                    # Clean up common artifacts
                    examiner = re.sub(r'\s+', ' ', examiner)
                    if len(examiner) > 2:  # Minimum reasonable name length
                        return examiner
            
            return None
            
        except Exception as e:
            logger.error(f"Examiner extraction failed: {e}")
            return None
    
    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender from text."""
        try:
            for pattern in self.gender_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    gender = matches[0].lower().strip()
                    # Normalize to standard values
                    if gender in ['männlich', 'male', 'm']:
                        return 'M'
                    elif gender in ['weiblich', 'female', 'f', 'w']:
                        return 'F'
            
            return None
            
        except Exception as e:
            logger.error(f"Gender extraction failed: {e}")
            return None
    
    def _parse_german_date(self, date_str: str) -> Optional[date]:
        """Parse German date format (DD.MM.YYYY) to date object."""
        try:
            # Use dateparser with German settings
            parsed = dateparser.parse(
                date_str,
                languages=['de'],
                settings={'DATE_ORDER': 'DMY'}
            )
            
            if parsed:
                return parsed.date()
            
            # Fallback: try manual parsing for DD.MM.YYYY
            if re.match(r'^\d{1,2}\.\d{1,2}\.\d{2,4}$', date_str):
                parts = date_str.split('.')
                if len(parts) == 3:
                    day, month, year = map(int, parts)
                    # Handle 2-digit years
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                    
                    return date(year, month, day)
            
            return None
            
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_str}': {e}")
            return None
    
    def is_sensitive_content(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if extracted metadata contains sensitive information.
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            True if metadata contains sensitive information
        """
        sensitive_fields = [
            'patient_first_name',
            'patient_last_name',
            'casenumber'
        ]
        
        # Check for non-empty sensitive fields
        for field in sensitive_fields:
            value = metadata.get(field)
            if value and value not in [None, '', 'Null', 'null', 'none', 'None', 'unknown', 'Unknown']:
                return True
        
        # Check for non-empty date of birth
        if metadata.get('patient_dob'):
            return True
        
        # Check for gender (if it's explicitly set)
        if metadata.get('patient_gender'):
            return True
        
        return False
    
    def merge_metadata(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge new metadata with existing, preferring non-empty values.
        
        Args:
            existing: Existing metadata dictionary
            new: New metadata to merge
            
        Returns:
            Merged metadata dictionary
        """
        merged = existing.copy()
        
        for key, value in new.items():
            if value and value not in [None, '', 'Unknown']:
                if not merged.get(key) or merged[key] in [None, '', 'Unknown']:
                    merged[key] = value
        
        return merged