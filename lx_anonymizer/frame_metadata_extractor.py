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
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

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
            r"Patient[:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)",
            r"Pat[\.:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)",
            r"Name[:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)",
            r"([A-Za-zäöüÄÖÜß]{2,})\s*,\s*([A-Za-zäöüÄÖÜß]{2,})",  # Last, First format
        ]

        self.dob_patterns = [
            r"geb[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"geboren[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"Geb\.Dat[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"DOB[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"(\d{1,2}\.\d{1,2}\.\d{2,4})",  # Standalone date
        ]

        self.case_patterns = [
            r"Fall[nr]*[\.:\s]*(\d+)",
            r"Case[:\s]*(\d+)",
            r"Fallnummer[:\s]*(\d+)",
            r"ID[:\s]*(\d+)",
        ]

        self.date_patterns = [
            r"Datum[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"Date[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"Untersuchung[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"(\d{1,2}\.\d{1,2}\.\d{2,4})",  # Any date format
        ]

        self.time_patterns = [
            r"Zeit[:\s]*(\d{1,2}:\d{2})",
            r"Time[:\s]*(\d{1,2}:\d{2})",
            r"(\d{1,2}:\d{2})",  # Any time format
        ]

        self.examiner_patterns = [
            r"Arzt[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",  # Limited length, word boundary
            r"Dr[\.:\s]+([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
            r"Untersucher[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
            r"Examiner[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
        ]

        self.gender_patterns = [
            r"(männlich|weiblich|male|female|m|f|w)",
        ]
        self.logger = logging.getLogger(__name__)

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
            "examiner_first_name": None,
            "examiner_last_name": None,
            "source": "frame_extraction",
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
            examiner_first, examiner_last = self._extract_examiner(text)
            if examiner_first:
                metadata["examiner_first_name"] = examiner_first
            if examiner_last:
                metadata["examiner_last_name"] = examiner_last

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
                        if "," in text and text.index(",") < text.index(name2):
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
                    if re.match(r"^\d{1,2}:\d{2}$", time_str):
                        return time_str

            return None

        except Exception as e:
            logger.error(f"Examination time extraction failed: {e}")
            return None

    def _extract_examiner(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract examiner first and last name from text."""
        try:
            for pattern in self.examiner_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    examiner = matches[0].strip()
                    examiner = re.sub(r"\s+", " ", examiner)
                    # Split examiner into first and last name
                    parts = examiner.split()
                    if len(parts) >= 2:
                        first_name, last_name = parts[0], " ".join(parts[1:])
                    else:
                        first_name, last_name = examiner, None
                    # Validate examiner candidate
                    if self._is_valid_examiner(examiner):
                        return first_name, last_name
                    else:
                        logger.debug(f"Rejected invalid examiner candidate: {examiner}")
            return None, None
        except Exception as e:
            logger.error(f"Examiner extraction failed: {e}")
            return None, None

    def _is_valid_examiner(self, examiner: str) -> bool:
        """
        Validate if a string is a plausible examiner name.

        Args:
            examiner: Candidate examiner name

        Returns:
            True if examiner appears valid, False if it looks like OCR garbage
        """
        if not examiner or not isinstance(examiner, str):
            return False

        # Strict validation for examiner names
        # Must be reasonable length and have proper word structure
        if len(examiner) < 3 or len(examiner) > 50:
            return False

        # Check for excessive special characters (sign of OCR garbage)
        special_char_count = sum(1 for c in examiner if c in ".-")
        total_chars = len(examiner)
        if total_chars > 0 and (special_char_count / total_chars) > 0.3:
            # More than 30% special characters = probably garbage
            logger.debug(
                f"Rejected examiner: too many special chars ({special_char_count}/{total_chars})"
            )
            return False

        # Check for reasonable word structure
        # Must have at least one proper word with >= 3 characters
        words = examiner.split()
        valid_words = [
            w
            for w in words
            if len(w) >= 3 and w.replace("-", "").replace(".", "").isalpha()
        ]

        # Require at least one substantial word (not just "Dr." or "M.")
        # and the majority of words should be valid
        if not valid_words or len(valid_words) < len(words) / 2:
            logger.debug(
                f"Rejected examiner: insufficient valid words (got {len(valid_words)}/{len(words)})"
            )
            return False

        # Additional check: ratio of single-character "words" indicates garbage
        single_char_words = sum(1 for w in words if len(w) == 1)
        if len(words) > 2 and single_char_words > len(words) / 2:
            logger.debug(
                f"Rejected examiner: too many single-char words ({single_char_words}/{len(words)})"
            )
            return False

        return True

    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender from text."""
        try:
            for pattern in self.gender_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    gender = matches[0].lower().strip()
                    # Normalize to standard values
                    if gender in ["männlich", "male", "m"]:
                        return "M"
                    elif gender in ["weiblich", "female", "f", "w"]:
                        return "F"

            return None

        except Exception as e:
            logger.error(f"Gender extraction failed: {e}")
            return None

    def _parse_german_date(self, date_str: str) -> Optional[date]:
        """Parse German date format (DD.MM.YYYY) to date object."""
        try:
            # Use dateparser with German settings
            parsed = dateparser.parse(
                date_str, languages=["de"], settings={"DATE_ORDER": "DMY"}
            )

            if parsed:
                return parsed.date()

            # Fallback: try manual parsing for DD.MM.YYYY
            if re.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$", date_str):
                parts = date_str.split(".")
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
        sensitive_fields = ["patient_first_name", "patient_last_name", "casenumber"]

        # Check for non-empty sensitive fields
        for field in sensitive_fields:
            value = metadata.get(field)
            if value and value not in [
                None,
                "",
                "Null",
                "null",
                "none",
                "None",
                "unknown",
                "Unknown",
            ]:
                return True

        # Check for non-empty date of birth
        if metadata.get("patient_dob"):
            return True

        # Check for gender (if it's explicitly set)
        if metadata.get("patient_gender"):
            return True

        return False

    _SENTINELS = {"unknown", "none", "n/a", "na", "unbekannt"}

    def _is_blank(self, v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            s = v.strip()
            return not s or s.lower() in self._SENTINELS
        return False  # keep 0, False, [], {} as valid

    def merge_metadata(
        self, existing: Dict[str, Any], new: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(existing or {})
        for k, nv in (new or {}).items():
            if self._is_blank(nv):
                continue
            cv = merged.get(k)

            # fill if missing/blank
            if self._is_blank(cv):
                merged[k] = nv
                continue

            if isinstance(cv, dict) and isinstance(nv, dict):
                merged[k] = self.merge_metadata(cv, nv)  # recursive
            elif isinstance(cv, list) and isinstance(nv, list):
                # simple union preserving order
                seen = set()
                out = []
                for x in cv + nv:
                    key = (type(x), str(x))
                    if key not in seen:
                        seen.add(key)
                        out.append(x)
                merged[k] = out
        self.logger.info(f"Merged metadata: {merged}")
        return merged
