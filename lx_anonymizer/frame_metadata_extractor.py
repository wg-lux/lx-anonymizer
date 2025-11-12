# lx_anonymizer/frame_cleaner/frame_metadata_extractor.py
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

from .sensitive_meta_interface import SensitiveMeta  # <<< integrate SensitiveMeta

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

    _SENTINELS = {"unknown", "none", "n/a", "na", "unbekannt", "undefined", "-"}

    def __init__(self):
        # runtime SensitiveMeta accumulator
        self.meta = SensitiveMeta()

        # Frame-specific patterns optimized for overlay text
        self.patient_patterns = [
            r"Patient[:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)",
            r"Pat[\.:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)",
            r"Name[:\s]*([A-Za-zäöüÄÖÜß]+)[\s,]*([A-Za-zäöüÄÖÜß]+)",
            r"([A-Za-zäöüÄÖÜß]{2,})\s*,\s*([A-Za-zäöüÄÖÜß]{2,})",  # Last, First
            r"\b([A-Z][a-zäöüß]{2,})\b",  # capitalized standalone
        ]

        self.dob_patterns = [
            r"geb[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"geboren[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"Geb\.Dat[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"DOB[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"(\d{10})",  # compact
            r"(\d{8})",
            r"(\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4})",
        ]

        self.case_patterns = [
            r"Fall[nr]*[\.:\s]*(\d+)",
            r"Case[:\s]*(\d+)",
            r"Fallnummer[:\s]*(\d+)",
            r"ID[:\s]*(\d+)",
            r"\b([A-Z]\s*\d{2,})\b",
        ]

        self.date_patterns = [
            r"Datum[:\s]*(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
            r"Date[:\s]*(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
            r"Untersuchung[:\s]*(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
            r"(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
        ]

        self.time_patterns = [
            r"Zeit[:\s]*(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
            r"Time[:\s]*(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
            r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
        ]

        self.examiner_patterns = [
            r"Arzt[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
            r"Dr[\.:\s]+([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
            r"Untersucher[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
            r"Examiner[:\s]*([A-Za-zäöüÄÖÜß\s\-\.]{3,50})(?:\s|$)",
        ]

        self.gender_patterns = [
            r"(männlich|weiblich|male|female|m|f|w)",
        ]

    # ---------- public API ----------

    def extract_metadata_from_frame_text(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from frame OCR text using specialized patterns.
        Writes through SensitiveMeta.safe_update, returns dict for compatibility.
        """
        if not text or not text.strip():
            return self.meta.to_dict()

        try:
            # names
            first_name, last_name = self._extract_patient_names(text)
            self.meta.safe_update(
                {
                    "patient_first_name": first_name,
                    "patient_last_name": last_name,
                }
            )

            # dob
            dob = self._extract_date_of_birth(text)
            self.meta.safe_update({"patient_dob": dob.isoformat() if isinstance(dob, date) else dob})

            # case number
            case_num = self._extract_case_number(text)
            self.meta.safe_update({"casenumber": case_num})

            # exam date/time
            exam_date = self._extract_examination_date(text)
            exam_time = self._extract_examination_time(text)
            self.meta.safe_update(
                {
                    "examination_date": exam_date.isoformat() if isinstance(exam_date, date) else exam_date,
                    "examination_time": exam_time,
                }
            )

            # examiner
            examiner_first, examiner_last = self._extract_examiner(text)
            self.meta.safe_update(
                {
                    "examiner_first_name": examiner_first,
                    "examiner_last_name": examiner_last,
                }
            )

            # gender
            gender = self._extract_gender(text)
            self.meta.safe_update({"patient_gender_name": gender})

            # mark source (won’t overwrite an existing non-blank)
            self.meta.safe_update({"center": None})  # no-op but illustrates safety
            return self.meta.to_dict()

        except Exception as e:
            logger.error(f"Frame metadata extraction failed: {e}")
            return self.meta.to_dict()

    def is_sensitive_content(self, metadata: Dict[str, Any]) -> bool:
        """Basic sensitive presence check (uses dict for call-site compatibility)."""
        sensitive_fields = ("patient_first_name", "patient_last_name", "casenumber", "patient_dob")
        for f in sensitive_fields:
            v = metadata.get(f)
            if self._is_nonblank(v):
                return True
        if self._is_nonblank(metadata.get("patient_gender_name")):
            return True
        return False

    def merge_metadata(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safe merge via SensitiveMeta:
        - Only non-blank values from `new` can fill blanks in `existing`
        - dicts/lists are kept as-is (SensitiveMeta is flat)
        - If both a DOB and an exam date are present but ambiguous:
          * newer date → examination_date
          * older date → patient_dob
        """
        # Start from existing, then apply new safely
        result_meta = SensitiveMeta.from_dict(existing or {})
        result_meta.safe_update(new or {})

        merged = result_meta.to_dict()

        # Enforce DOB vs exam-date rule if both (or ambiguous) are present anywhere
        dob_raw = (new or {}).get("patient_dob") or (existing or {}).get("patient_dob")
        exam_raw = (new or {}).get("examination_date") or (existing or {}).get("examination_date")

        # If one of them is missing but we have multiple date-like values somewhere else,
        # try to infer: choose max as exam, min as dob.
        if dob_raw or exam_raw:
            dob_dt = self._to_date(dob_raw)
            exam_dt = self._to_date(exam_raw)

            if dob_dt and exam_dt:
                # both parse → order
                if exam_dt < dob_dt:
                    exam_dt, dob_dt = dob_dt, exam_dt
            else:
                # if one is missing, do nothing special
                pass

            # write back safely (isoformat)
            inferred = {}
            if exam_dt:
                inferred["examination_date"] = exam_dt.isoformat()
            if dob_dt:
                inferred["patient_dob"] = dob_dt.isoformat()
            result_meta.safe_update(inferred)
            merged = result_meta.to_dict()

        logger.info(f"Merged metadata: {merged}")
        return merged

    # ---------- helpers ----------

    @staticmethod
    def _is_nonblank(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            s = v.strip()
            return bool(s) and s.lower() not in FrameMetadataExtractor._SENTINELS
        return True  # keep 0/False/[]/{} as “present” for our purpose

    def _to_date(self, v: Any) -> Optional[date]:
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str) and v.strip():
            # try ISO first, then German
            try:
                # YYYY-MM-DD
                return datetime.strptime(v.strip(), "%Y-%m-%d").date()
            except Exception:
                parsed = dateparser.parse(v.strip(), languages=["de"], settings={"DATE_ORDER": "DMY"})
                if parsed:
                    return parsed.date()
        return None

    def _extract_patient_names(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            for pattern in self.patient_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if not matches:
                    continue
                m0 = matches[0]
                if isinstance(m0, tuple) and len(m0) >= 2:
                    name1, name2 = m0[:2]
                    name1, name2 = name1.strip(), name2.strip()
                    # Heuristic: "Last, First" if comma appears before name2
                    if "," in text and text.index(",") < text.index(name2):
                        return name2, name1
                    return name1, name2
                if isinstance(m0, str):
                    return m0.strip(), None
            return None, None
        except Exception as e:
            logger.error(f"Patient name extraction failed: {e}")
            return None, None

    def _extract_date_of_birth(self, text: str) -> Optional[date]:
        try:
            # 1) labeled
            for pattern in self.dob_patterns[:4]:
                m = re.findall(pattern, text, re.IGNORECASE)
                if m:
                    d = self._parse_german_date(m[0].strip())
                    if d:
                        return d
            # 2) compact
            for pattern in self.dob_patterns[4:6]:
                m = re.findall(pattern, text)
                if m:
                    d = self._parse_compact_date(m[0].strip())
                    if d:
                        return d
            # 3) separators
            m = re.findall(self.dob_patterns[6], text)
            if m:
                d = self._parse_german_date(m[0].strip())
                if d:
                    return d
            return None
        except Exception as e:
            logger.error(f"DOB extraction failed: {e}")
            return None

    def _parse_compact_date(self, date_str: str) -> Optional[date]:
        try:
            digits = re.sub(r"\D", "", date_str)
            if len(digits) == 8:
                day = int(digits[0:2]); month = int(digits[2:4]); year = int(digits[4:8])
                if 1 <= day <= 31 and 1 <= month <= 12:
                    return date(year, month, day)
            elif len(digits) == 10:
                day = int(digits[0:2]); month = int(digits[2:4]); year = int(digits[4:8])
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                    return date(year, month, day)
            return None
        except (ValueError, IndexError) as e:
            logger.debug(f"Compact date parsing failed for '{date_str}': {e}")
            return None

    def _extract_case_number(self, text: str) -> Optional[str]:
        try:
            for pattern in self.case_patterns:
                m = re.findall(pattern, text, re.IGNORECASE)
                if m:
                    case = re.sub(r"\s+", " ", m[0].strip())
                    return case
            return None
        except Exception as e:
            logger.error(f"Case number extraction failed: {e}")
            return None

    def _extract_examination_date(self, text: str) -> Optional[date]:
        try:
            for pattern in self.date_patterns:
                m = re.findall(pattern, text, re.IGNORECASE)
                if m:
                    d = self._parse_german_date(m[0].strip())
                    if d:
                        return d
            return None
        except Exception as e:
            logger.error(f"Examination date extraction failed: {e}")
            return None

    def _extract_examination_time(self, text: str) -> Optional[str]:
        try:
            for pattern in self.time_patterns:
                m = re.findall(pattern, text, re.IGNORECASE)
                if m:
                    t = m[0].strip()
                    if re.match(r"^\d{1,2}:\d{2}$", t):
                        return t
            return None
        except Exception as e:
            logger.error(f"Examination time extraction failed: {e}")
            return None

    def _extract_examiner(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            for pattern in self.examiner_patterns:
                m = re.findall(pattern, text, re.IGNORECASE)
                if not m:
                    continue
                examiner = re.sub(r"\s+", " ", m[0].strip())
                parts = examiner.split()
                if len(parts) >= 2:
                    first_name, last_name = parts[0], " ".join(parts[1:])
                else:
                    first_name, last_name = examiner, None
                if self._is_valid_examiner(examiner):
                    return first_name, last_name
                logger.debug(f"Rejected invalid examiner candidate: {examiner}")
            return None, None
        except Exception as e:
            logger.error(f"Examiner extraction failed: {e}")
            return None, None

    def _is_valid_examiner(self, examiner: str) -> bool:
        if not isinstance(examiner, str) or not examiner:
            return False
        if len(examiner) < 3 or len(examiner) > 50:
            return False
        special = sum(1 for c in examiner if c in ".-")
        if len(examiner) and special / len(examiner) > 0.3:
            return False
        words = examiner.split()
        valid_words = [w for w in words if len(w) >= 3 and w.replace("-", "").replace(".", "").isalpha()]
        if not valid_words or len(valid_words) < len(words) / 2:
            return False
        single_char_words = sum(1 for w in words if len(w) == 1)
        if len(words) > 2 and single_char_words > len(words) / 2:
            return False
        return True

    def _extract_gender(self, text: str) -> Optional[str]:
        try:
            for pattern in self.gender_patterns:
                m = re.findall(pattern, text, re.IGNORECASE)
                if m:
                    g = m[0].lower().strip()
                    if g in ("männlich", "male", "m"):
                        return "M"
                    if g in ("weiblich", "female", "f", "w"):
                        return "F"
            return None
        except Exception as e:
            logger.error(f"Gender extraction failed: {e}")
            return None

    def _parse_german_date(self, date_str: str) -> Optional[date]:
        try:
            normalized = re.sub(r"\s+", "", date_str).replace("/", ".").replace("-", ".")
            parsed = dateparser.parse(normalized, languages=["de"], settings={"DATE_ORDER": "DMY"})
            if parsed:
                return parsed.date()
            if re.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$", normalized):
                d, m, y = map(int, normalized.split("."))
                if y < 100:
                    y += 2000 if y < 50 else 1900
                if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                    return date(y, m, d)
            return None
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_str}': {e}")
            return None
