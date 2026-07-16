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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional, Tuple, cast

import dateparser  # type: ignore[import-untyped]

from lx_anonymizer.regex_patterns import (
    FRAME_CASE_PATTERNS,
    FRAME_DATE_PATTERNS,
    FRAME_DOB_PATTERNS,
    FRAME_EXAMINER_PATTERNS,
    FRAME_GENDER_PATTERNS,
    FRAME_PATIENT_PATTERNS,
    FRAME_TIME_PATTERNS,
    DATE_DOT_FLEX_RE,
    MULTISPACE_RE,
    NON_DIGIT_RE,
    TIME_HH_MM_RE,
)
from lx_anonymizer.sensitive_meta_interface import (
    SensitiveMeta,
)  # <<< integrate SensitiveMeta

logger = logging.getLogger(__name__)


class _DateRole(str, Enum):
    DOB = "dob"
    EXAMINATION = "examination_date"


@dataclass(frozen=True)
class _FrameDateCandidate:
    value: date
    start: int
    end: int
    role: _DateRole | None
    followed_by_time: bool


_FRAME_DATE_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\d{1,2}\s*(?P<separator>[.\-/])\s*\d{1,2}\s*"
    r"(?P=separator)\s*\d{2,4}"
    r"|\d{1,2}\s+\d{1,2}\s+\d{2,4}"
    r")(?!\d)"
)
_DOB_LABEL_BEFORE_DATE_RE = re.compile(
    r"(?:\bDOB|\bgeb(?:oren)?|Geb\.Dat(?:um)?|Geburtsdatum)\s*[.:\-]?\s*$",
    re.IGNORECASE,
)
_EXAM_LABEL_BEFORE_DATE_RE = re.compile(
    r"(?:\bDatum|\bDate|\bUntersuchung|\bExam(?:ination)?)\s*[.:\-]?\s*$",
    re.IGNORECASE,
)
_TIME_AFTER_DATE_RE = re.compile(r"^\s+\d{1,2}[:.]\d{2}(?:[:.]\d{2})?")


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

    def __init__(self) -> None:
        # runtime SensitiveMeta accumulator
        self.meta = SensitiveMeta()

        # Frame-specific patterns optimized for overlay text
        self.patient_patterns: list[str] = list(FRAME_PATIENT_PATTERNS)
        self.dob_patterns: list[str] = list(FRAME_DOB_PATTERNS)
        self.case_patterns: list[str] = list(FRAME_CASE_PATTERNS)
        self.date_patterns: list[str] = list(FRAME_DATE_PATTERNS)
        self.time_patterns: list[str] = list(FRAME_TIME_PATTERNS)
        self.examiner_patterns: list[str] = list(FRAME_EXAMINER_PATTERNS)
        self.gender_patterns: list[str] = list(FRAME_GENDER_PATTERNS)

    # ---------- public API ----------

    def extract_metadata_from_frame_text(self, text: str) -> dict[str, object]:
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
                    "first_name": first_name,
                    "last_name": last_name,
                }
            )

            # Resolve date roles together. Independent first-match extraction can
            # otherwise assign the first overlay date to both DOB and exam date.
            dob, exam_date = self._resolve_frame_dates(text)
            self.meta.safe_update(
                {"dob": dob.isoformat() if isinstance(dob, date) else dob}
            )

            # case number
            case_num = self._extract_case_number(text)
            self.meta.safe_update({"casenumber": case_num})

            # exam date/time
            exam_time = self._extract_examination_time(text)
            self.meta.safe_update(
                {
                    "examination_date": exam_date.isoformat()
                    if isinstance(exam_date, date)
                    else exam_date,
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
            self.meta.safe_update({"gender": gender})

            # mark source (won’t overwrite an existing non-blank)
            self.meta.safe_update({"center": None})  # no-op but illustrates safety
            return self.meta.to_dict()

        except Exception as e:
            logger.error(f"Frame metadata extraction failed: {e}")
            return self.meta.to_dict()

    def is_sensitive_content(self, metadata: Mapping[str, object]) -> bool:
        """Basic sensitive presence check (uses dict for call-site compatibility)."""
        sensitive_fields = (
            "first_name",
            "last_name",
            "casenumber",
            "dob",
        )
        for f in sensitive_fields:
            v = metadata.get(f)
            if self._is_nonblank(v):
                return True
        if self._is_nonblank(metadata.get("gender")):
            return True
        return False

    def is_complete(self, metadata: Mapping[str, object]) -> bool:
        """
        Returns True when enough high-signal identifiers are present.
        Used to stop early when smart sampling is enabled.
        """
        if not metadata:
            return False

        has_first = self._is_nonblank(metadata.get("first_name"))
        has_last = self._is_nonblank(metadata.get("last_name"))
        has_dob = self._is_nonblank(metadata.get("dob"))
        has_case = self._is_nonblank(metadata.get("casenumber"))
        has_exam_date = self._is_nonblank(metadata.get("examination_date"))

        # Strong completion: full name + DOB
        if has_first and has_last and has_dob:
            return True

        # Alternative: full name + case + exam date
        if has_first and has_last and has_case and has_exam_date:
            return True

        # Allow case+DOB with partial name as fallback
        if (has_first or has_last) and has_dob and has_case:
            return True

        return False

    def merge_metadata(
        self, existing: Mapping[str, object], new: Mapping[str, object]
    ) -> dict[str, object]:
        """
        Safe merge via SensitiveMeta:
        - Only non-blank values from `new` can fill blanks in `existing`
        - dicts/lists are kept as-is (SensitiveMeta is flat)
        - If both a DOB and an exam date are present but ambiguous:
          * newer date → examination_date
          * older date → dob
        """
        # Start from existing, then apply new safely
        result_meta = SensitiveMeta.from_dict(existing or {})
        result_meta.safe_update(new or {})

        merged = result_meta.to_dict()

        # Enforce DOB vs exam-date rule if both (or ambiguous) are present anywhere
        dob_raw = (new or {}).get("dob") or (existing or {}).get("dob")
        exam_raw = (new or {}).get("examination_date") or (existing or {}).get(
            "examination_date"
        )

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
            inferred: dict[str, object] = {}
            if exam_dt:
                inferred["examination_date"] = exam_dt.isoformat()
            if dob_dt:
                inferred["dob"] = dob_dt.isoformat()
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
            except ValueError:
                parsed = dateparser.parse(
                    v.strip(), languages=["de"], settings={"DATE_ORDER": "DMY"}
                )
                if parsed:
                    return parsed.date()
        return None

    def _extract_patient_names(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            for pattern in self.patient_patterns:
                matches = cast(
                    list[str | tuple[str, ...]],
                    re.findall(pattern, text, re.IGNORECASE),
                )
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

    def _resolve_frame_dates(self, text: str) -> tuple[date | None, date | None]:
        """Resolve DOB and examination date from one shared candidate set.

        Explicit labels take precedence. A date immediately followed by a time is
        treated as an examination timestamp. If an overlay contains multiple
        otherwise-unlabelled dates, the oldest distinct date is the DOB and the
        newest is the examination date. One candidate is never assigned to both
        roles.
        """
        candidates = self._collect_frame_date_candidates(text)
        if not candidates:
            return None, None

        dob = self._first_candidate_for_role(candidates, _DateRole.DOB)
        examination = self._first_candidate_for_role(candidates, _DateRole.EXAMINATION)
        if examination is None:
            examination = next(
                (candidate for candidate in candidates if candidate.followed_by_time),
                None,
            )

        distinct_dates = sorted({candidate.value for candidate in candidates})
        if dob is None:
            dob = next(
                (
                    candidate
                    for value in distinct_dates
                    if examination is None or value != examination.value
                    for candidate in candidates
                    if candidate.value == value
                ),
                None,
            )
        if examination is None:
            examination = next(
                (
                    candidate
                    for value in reversed(distinct_dates)
                    if dob is None or value != dob.value
                    for candidate in candidates
                    if candidate.value == value
                ),
                None,
            )

        if (
            dob is not None
            and examination is not None
            and dob.value == examination.value
        ):
            # Preserve the explicitly supported role instead of duplicating an
            # ambiguous single date into two metadata fields.
            if dob.role is _DateRole.DOB:
                examination = None
            else:
                dob = None

        if (
            dob is not None
            and examination is not None
            and dob.value > examination.value
        ):
            dob, examination = examination, dob

        return (
            dob.value if dob is not None else None,
            examination.value if examination is not None else None,
        )

    def _collect_frame_date_candidates(
        self, text: str
    ) -> tuple[_FrameDateCandidate, ...]:
        candidates: list[_FrameDateCandidate] = []
        for match in _FRAME_DATE_CANDIDATE_RE.finditer(text):
            parsed = self._parse_german_date(match.group(0))
            if parsed is None:
                continue
            prefix = text[max(0, match.start() - 32) : match.start()]
            suffix = text[match.end() : match.end() + 16]
            role: _DateRole | None = None
            if _DOB_LABEL_BEFORE_DATE_RE.search(prefix):
                role = _DateRole.DOB
            elif _EXAM_LABEL_BEFORE_DATE_RE.search(prefix):
                role = _DateRole.EXAMINATION
            candidates.append(
                _FrameDateCandidate(
                    value=parsed,
                    start=match.start(),
                    end=match.end(),
                    role=role,
                    followed_by_time=bool(_TIME_AFTER_DATE_RE.match(suffix)),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _first_candidate_for_role(
        candidates: tuple[_FrameDateCandidate, ...], role: _DateRole
    ) -> _FrameDateCandidate | None:
        return next(
            (candidate for candidate in candidates if candidate.role is role), None
        )

    def _parse_compact_date(self, date_str: str) -> Optional[date]:
        try:
            digits = NON_DIGIT_RE.sub("", date_str)
            if len(digits) == 8:
                day = int(digits[0:2])
                month = int(digits[2:4])
                year = int(digits[4:8])
                if 1 <= day <= 31 and 1 <= month <= 12:
                    return date(year, month, day)
            elif len(digits) == 10:
                day = int(digits[0:2])
                month = int(digits[2:4])
                year = int(digits[4:8])
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
                    case = MULTISPACE_RE.sub(" ", m[0].strip())
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
                    if TIME_HH_MM_RE.match(t):
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
                examiner = MULTISPACE_RE.sub(" ", m[0].strip())
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
        if not examiner:
            return False
        if len(examiner) < 3 or len(examiner) > 50:
            return False
        special = sum(1 for c in examiner if c in ".-")
        if len(examiner) and special / len(examiner) > 0.3:
            return False
        words = examiner.split()
        valid_words = [
            w
            for w in words
            if len(w) >= 3 and w.replace("-", "").replace(".", "").isalpha()
        ]
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
            normalized = re.sub(
                r"\s*[.\-/]\s*|\s+",
                ".",
                date_str.strip(),
            )
            parsed = dateparser.parse(
                normalized, languages=["de"], settings={"DATE_ORDER": "DMY"}
            )
            if parsed:
                return parsed.date()
            if DATE_DOT_FLEX_RE.match(normalized):
                d, m, y = map(int, normalized.split("."))
                if y < 100:
                    y += 2000 if y < 50 else 1900
                if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                    return date(y, m, d)
            return None
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_str}': {e}")
            return None
