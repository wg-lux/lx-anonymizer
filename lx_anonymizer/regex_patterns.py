"""Central regex patterns shared across OCR, NER, report, and video helpers."""

from __future__ import annotations

import re
from typing import Final, Pattern

GERMAN_NAME_CHARS: Final[str] = "A-Za-zäöüÄÖÜß"
GERMAN_WORD_CHARS: Final[str] = "A-Za-zÄÖÜäöüß"

GERMAN_WORD_RE: Final[Pattern[str]] = re.compile(rf"[{GERMAN_WORD_CHARS}]{{2,}}")
REPEATED_CHAR_RE: Final[Pattern[str]] = re.compile(r"(.)\1{3,}")
MULTISPACE_RE: Final[Pattern[str]] = re.compile(r"\s+")
MULTISPACE_2PLUS_RE: Final[Pattern[str]] = re.compile(r"\s{2,}")
NON_ALNUM_COMPACT_RE: Final[Pattern[str]] = re.compile(r"[^a-z0-9]")
NON_ALNUM_SPACE_RE: Final[Pattern[str]] = re.compile(r"[^a-z0-9]")
NON_DIGIT_RE: Final[Pattern[str]] = re.compile(r"\D")

DATE_8_DIGIT_RE: Final[Pattern[str]] = re.compile(r"\d{8}")
DATE_DOT_FULL_RE: Final[Pattern[str]] = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")
DATE_DOT_FLEX_RE: Final[Pattern[str]] = re.compile(
    r"^\d{1,2}\.\d{1,2}\.\d{2,4}$"
)
DATE_TEXT_RE: Final[Pattern[str]] = re.compile(
    r"(\d{1,2}[.\s]?\d{1,2}[.\s]?\d{2,4})|(\d{8})"
)
DATE_OVERLAY_PATTERN: Final[str] = (
    r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|"
    r"\d{1,2}[-./]\d{1,2}[-./]\d{4}"
)
TIME_OVERLAY_PATTERN: Final[str] = r"\d{1,2}:\d{2}(?::\d{2})?"
TIME_HH_MM_RE: Final[Pattern[str]] = re.compile(r"^\d{1,2}:\d{2}$")

CASE_OVERLAY_PATTERN: Final[str] = r"[A-Z]\s*\d{4,}/\d{4}"
COMPACT_CODE_PATTERN: Final[str] = r"\b[A-Z]\s*\d{5,}\b|\b[A-Z]\d{5,}\b"
DEVICE_ID_PATTERN: Final[str] = r"\d{8,}"
RATIO_PATTERN: Final[str] = r"\b\d+(?:[.,]\d+)?/\d+(?:[.,]\d+)?\b"

STRUCTURED_OVERLAY_PATTERNS: Final[tuple[str, ...]] = (
    TIME_OVERLAY_PATTERN,
    DATE_OVERLAY_PATTERN,
    CASE_OVERLAY_PATTERN,
    COMPACT_CODE_PATTERN,
    DEVICE_ID_PATTERN,
    RATIO_PATTERN,
)
STRUCTURED_OVERLAY_RE: Final[Pattern[str]] = re.compile(
    "|".join(f"(?:{pattern})" for pattern in STRUCTURED_OVERLAY_PATTERNS)
)
STRUCTURED_OVERLAY_LOOSE_RE: Final[Pattern[str]] = re.compile(
    "|".join(
        f"(?:{pattern})"
        for pattern in (
            TIME_OVERLAY_PATTERN,
            COMPACT_CODE_PATTERN,
            RATIO_PATTERN,
            r"\b\d{4,}\b",
        )
    )
)

PATIENT_LINE_RE: Final[Pattern[str]] = re.compile(
    r"pat(?:ient|ientin|\.|iont|bien)", re.IGNORECASE
)
EXAMINER_LINE_RE: Final[Pattern[str]] = re.compile(r"unters\W*arzt", re.IGNORECASE)
EXAMINATION_LINE_RE: Final[Pattern[str]] = re.compile(
    r"unters\.:|u-datum:|eingang\s*am:", re.IGNORECASE
)
REPORT_ENTRY_DATE_RE: Final[Pattern[str]] = re.compile(
    r"Eingang am:\s*(\d{2}\.\d{2}\.\d{4})"
)

LLM_TITLE_TOKEN_RE: Final[Pattern[str]] = re.compile(
    r"\b(?:herrn?|frau|Herr||fru|monsieur|madame|dr\.?|prof\.?|professor|ing\.?)\b",
    re.IGNORECASE,
)
LLM_AGE_TOKEN_RE: Final[Pattern[str]] = re.compile(r"\b\d{1,3}\s*jahre?\b", re.IGNORECASE)
LLM_NARRATIVE_TOKEN_RE: Final[Pattern[str]] = re.compile(
    r"\b(?:befund|patient|screening|beschwerden|koloskopie|gastroskopie)\b",
    re.IGNORECASE,
)

MEDICAL_QUALITY_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"Patient|Untersuchung|Diagnose|Befund|Behandlung", re.IGNORECASE),
    re.compile(r"mm|cm|Grad|°|%", re.IGNORECASE),
    re.compile(r"links|rechts|lateral|medial|anterior|posterior", re.IGNORECASE),
    re.compile(r"Jahr|Jahre|Tag|Tage|Monat|Monate", re.IGNORECASE),
)

OCR_ALLOWED_TEXT_RE: Final[Pattern[str]] = re.compile(r"[^\w\s.,:;/-ÄÖÜäöüß]")
REPEATED_PUNCT_RE: Final[Pattern[str]] = re.compile(r"([.,:;])\1{1,}")
PUNCT_RUN_RE: Final[Pattern[str]] = re.compile(r"[.,:;]{2,}")

ISO_DATE_RE: Final[Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LONG_NUMBER_RE: Final[Pattern[str]] = re.compile(r"\b\d{5,}\b")

FRAME_PATIENT_PATTERNS: Final[tuple[str, ...]] = (
    rf"Patient[:\s]*([{GERMAN_NAME_CHARS}]+)[\s,]*([{GERMAN_NAME_CHARS}]+)",
    rf"Pat[\.:\s]*([{GERMAN_NAME_CHARS}]+)[\s,]*([{GERMAN_NAME_CHARS}]+)",
    rf"Name[:\s]*([{GERMAN_NAME_CHARS}]+)[\s,]*([{GERMAN_NAME_CHARS}]+)",
    rf"([{GERMAN_NAME_CHARS}]{{2,}})\s*,\s*([{GERMAN_NAME_CHARS}]{{2,}})",
    r"\b([A-Z][a-zäöüß]{2,})\b",
)
FRAME_DOB_PATTERNS: Final[tuple[str, ...]] = (
    r"geb[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
    r"geboren[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
    r"Geb\.Dat[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
    r"DOB[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})",
    r"(\d{10})",
    r"(\d{8})",
    r"(\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4})",
)
FRAME_CASE_PATTERNS: Final[tuple[str, ...]] = (
    r"Fall[nr]*[\.:\s]*(\d+)",
    r"Case[:\s]*(\d+)",
    r"Fallnummer[:\s]*(\d+)",
    r"ID[:\s]*(\d+)",
    r"\b([A-Z]\s*\d{2,})\b",
)
FRAME_DATE_PATTERNS: Final[tuple[str, ...]] = (
    r"Datum[:\s]*(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
    r"Date[:\s]*(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
    r"Untersuchung[:\s]*(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
    r"(\d{1,2}[\.\-/\s]+\d{1,2}[\.\-/\s]+\d{2,4})",
)
FRAME_TIME_PATTERNS: Final[tuple[str, ...]] = (
    r"Zeit[:\s]*(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
    r"Time[:\s]*(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
    r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)",
)
FRAME_EXAMINER_PATTERNS: Final[tuple[str, ...]] = (
    rf"Arzt[:\s]*([{GERMAN_NAME_CHARS}\s\-\.]{{3,50}})(?:\s|$)",
    rf"Dr[\.:\s]+([{GERMAN_NAME_CHARS}\s\-\.]{{3,50}})(?:\s|$)",
    rf"Untersucher[:\s]*([{GERMAN_NAME_CHARS}\s\-\.]{{3,50}})(?:\s|$)",
    rf"Examiner[:\s]*([{GERMAN_NAME_CHARS}\s\-\.]{{3,50}})(?:\s|$)",
)
FRAME_GENDER_PATTERNS: Final[tuple[str, ...]] = (
    r"(männlich|weiblich|male|female|m|f|w)",
)

FALLBACK_PATIENT_FULL_RE: Final[Pattern[str]] = re.compile(
    rf"(?:Patient|Pat|Patientin|Pat\.):?\s*"
    rf"([{GERMAN_NAME_CHARS}\-]+)[,\s]+([{GERMAN_NAME_CHARS}\-]+)\s+"
    r"(?:geb\.|geboren am|Geb\.Dat\.|geboren):?\s*"
    r"(\d{1,2}\.\d{1,2}\.\d{4})"
    r"(?:.*?(?:Fallnummer|Fallnr\.|Fall\.Nr\.|Fall-Nr):?\s*(\d+))?",
    re.IGNORECASE,
)
FALLBACK_PATIENT_NAME_RE: Final[Pattern[str]] = re.compile(
    rf"(?:Patient|Pat|Patientin|Pat\.):?\s*"
    rf"([{GERMAN_NAME_CHARS}\-]+)[,\s]+([{GERMAN_NAME_CHARS}\-]+)",
    re.IGNORECASE,
)
FALLBACK_PATIENT_TITLED_NAME_RE: Final[Pattern[str]] = re.compile(
    rf"(?:Patient|Pat|Patientin|Pat\.):?\s*"
    rf"((?:Dr\.|Prof\.|Herr|Frau)?\s*[{GERMAN_NAME_CHARS}\-\s]+)",
    re.IGNORECASE,
)
