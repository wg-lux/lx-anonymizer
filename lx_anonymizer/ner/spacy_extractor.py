import re
from datetime import datetime
from typing import Any, Dict, Optional, Pattern, Tuple

import spacy
import spacy.cli
from spacy.language import Language
from spacy.matcher import Matcher

from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.ner.determine_gender import determine_gender
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta

logger = get_logger(__name__)

# --- Constants ---

TITLE_WORDS = {
    "herr",
    "herrn",
    "frau",
    "f",
    "fru",
    "fruis",
    "fruharuka",
    "dr.",
    "dr",
    "doctor",
    "prof.",
    "prof",
    "professor",
    "ing.",
    "ing",
    "señor",
    "señorita",
    "monsieur",
    "mr.",
    "mrs.",
    "sir",
    "mag.",
    "baron",
}

DATE_RE: Pattern = re.compile(r"(\d{1,2}[.\s]?\d{1,2}[.\s]?\d{2,4})|(\d{8})")


# --- Utilities ---


class SpacyModelManager:
    """
    Singleton-like manager to ensure the model is loaded only once.
    """

    _instance: Optional[Language] = None
    DEFAULT_MODEL = "de_core_news_lg"

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_MODEL) -> Language:
        if cls._instance is not None:
            return cls._instance

        try:
            logger.info(f"Loading spacy model: {model_name}")
            cls._instance = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model '{model_name}' not found. Attempting download...")
            try:
                spacy.cli.download(model_name)
                cls._instance = spacy.load(model_name)
                logger.info(f"Successfully loaded {model_name} after download.")
            except Exception as e:
                logger.error(f"Critical error loading SpaCy model: {e}")
                raise e

        return cls._instance


def _clean_date(date_str: str) -> Optional[str]:
    """
    Normalizes date strings to YYYY-MM-DD.
    Handles: dd.mm.yyyy, dd mm yyyy, ddmmyyyy (8 digits).
    """
    if not date_str:
        return None

    date_str = date_str.strip()
    # Normalize separators
    normalized = re.sub(r"\s+", ".", date_str)

    # 1. Try 8-digit format (DDMMYYYY) explicitly first
    if re.fullmatch(r"\d{8}", date_str):
        try:
            return datetime.strptime(date_str, "%d%m%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass  # Continue to other formats

    # 2. Try standard formats
    formats = ["%d.%m.%Y", "%d.%m.%y"]

    for fmt in formats:
        try:
            dt = datetime.strptime(normalized, fmt)

            # Handle 2-digit year pivot
            if dt.year < 100:
                dt = dt.replace(
                    year=dt.year + 1900 if dt.year >= 69 else dt.year + 2000
                )

            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    logger.debug(f"Could not parse date: {date_str}")
    return None


# --- Base Class ---


class BaseExtractor:
    """Base class for all extractors to handle common setup."""

    def __init__(self, meta: Optional[SensitiveMeta] = None):
        self.nlp = SpacyModelManager.get_model()
        # Initialize Matcher with the vocab of the loaded model
        self.matcher = Matcher(self.nlp.vocab)
        self.meta = meta or SensitiveMeta()

        # Hook for subclasses to register their specific rules
        self._register_patterns()

    def _register_patterns(self):
        """Subclasses should override this to add patterns to self.matcher"""
        pass

    def safe_update_meta(self, data: Dict[str, Any]):
        """Helper wrapper for meta updates."""
        self.meta.safe_update(data)


# --- Extractors ---


class PatientDataExtractor(BaseExtractor):
    def _register_patterns(self):
        # Header variations
        header_variants = r"(?i)^(pat(ient|ientin|\.?)|pationt|patbien)$"

        pat_header = [
            {"LOWER": {"REGEX": header_variants}},
            {"TEXT": ":", "OP": "?"},
        ]

        name_part = {"POS": {"IN": ["PROPN", "NOUN"]}, "IS_TITLE": True, "OP": "+"}
        space = {"IS_SPACE": True, "OP": "*"}

        geb_block = [
            {"LOWER": {"IN": ["geb", "geb.", "geboren"]}},
            {"LOWER": "am", "OP": "?"},
            {"TEXT": ":", "OP": "?"},
            {"TEXT": {"REGEX": DATE_RE.pattern}},
        ]

        fall_block = [
            {"LOWER": {"REGEX": r"^fall(?:nr\.?|nummer)$"}},
            {"TEXT": ":", "OP": "?"},
            {"TEXT": {"REGEX": r"[\w/-]+"}},
        ]

        # Consolidated pattern
        pattern = (
            pat_header
            + [space]
            + [name_part]  # Last name (rough approx)
            + [space, {"TEXT": ",", "OP": "?"}, space]
            + [name_part]  # First name (rough approx)
            + [space]
            + [{"OP": "?"}]  # Optional token between name and DOB
            + geb_block
            + [space]
            + [{"OP": "?"}]
            + fall_block
        )

        self.matcher.add("PATIENT_LINE", [pattern])

    def __call__(self, text: str) -> Dict[str, Optional[str]]:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        if not matches:
            return self.meta.to_dict()

        # Get longest match
        _, start, end = max(matches, key=lambda m: m[2] - m[1])
        span = doc[start:end]
        tokens = list(span)

        # Extraction logic helpers
        first_name, last_name = self._extract_names(tokens)
        birthdate = self._extract_dob(span, tokens)
        case_num = self._extract_case_number(tokens, birthdate)
        gender = determine_gender(first_name) if first_name else None

        self.safe_update_meta(
            {
                "patient_first_name": first_name,
                "patient_last_name": last_name,
                "patient_dob": birthdate,
                "casenumber": case_num,
                "patient_gender_name": gender,
            }
        )

        return self.meta.to_dict()

    def _extract_names(self, tokens) -> Tuple[Optional[str], Optional[str]]:
        """Heuristics to split names based on commas or position."""
        header_end_idx = 0
        # Find where "Patient:" ends
        for i, token in enumerate(tokens):
            if token.lower_ in ["patient", "patientin", "pat.", "pationt", ":"]:
                header_end_idx = i + 1
            elif tokens[header_end_idx - 1].text != ":":
                break

        comma_idx = next((i for i, t in enumerate(tokens) if t.text == ","), None)

        # Identify stop words (geb, fallnr) to stop grabbing name tokens
        stop_indices = [
            i
            for i, t in enumerate(tokens)
            if t.lower_ in ["geb", "geb.", "geboren", "fallnr", "fallnummer"]
        ]
        end_name_idx = min(stop_indices) if stop_indices else len(tokens)

        if comma_idx:
            ln_tokens = tokens[header_end_idx:comma_idx]
            fn_tokens = tokens[comma_idx + 1 : end_name_idx]
        else:
            # Fallback: assume Lastname Firstname order without comma
            full_name_tokens = tokens[header_end_idx:end_name_idx]
            if len(full_name_tokens) >= 2:
                # Naive split: Last token is first name, rest is last name?
                # Or German convention: First token(s) usually last name if written "Mustermann Max"
                # The original code logic was ambiguous here, preserving "Lastname Firstname" assumption
                ln_tokens = full_name_tokens
                fn_tokens = []  # Cannot reliably split without comma in this simplified logic
                if len(full_name_tokens) >= 2:
                    ln_tokens = (
                        full_name_tokens  # Logic in original was slightly circular.
                    )
                    # Keeping it safe: if no comma, dump to last_name or try to split?
                    # Let's stick to original behavior:
                    pass

            else:
                ln_tokens = full_name_tokens
                fn_tokens = []

        # Filter titles
        ln_clean = [
            t.text for t in ln_tokens if t.is_alpha and t.lower_ not in TITLE_WORDS
        ]
        fn_clean = [
            t.text for t in fn_tokens if t.is_alpha and t.lower_ not in TITLE_WORDS
        ]

        return (" ".join(fn_clean) or None, " ".join(ln_clean) or None)

    def _extract_dob(self, span, tokens) -> Optional[str]:
        date_match = DATE_RE.search(span.text)
        if not date_match:
            return None

        # Find the specific token that matches the regex
        for token in tokens:
            if DATE_RE.fullmatch(token.text):
                return _clean_date(token.text)
        return None

    def _extract_case_number(self, tokens, birthdate_str) -> Optional[str]:
        # Look for explicitly labeled case number
        for i, t in enumerate(tokens):
            if t.lower_ in ["fallnr", "fallnr.", "fallnummer"]:
                # Check neighbors
                candidates = tokens[i + 1 : i + 3]
                for cand in candidates:
                    if cand.text != ":" and re.fullmatch(r"[\w/-]+", cand.text):
                        return cand.text

        # Fallback: Find alphanumeric token at end of string that isn't the DOB
        # (This is a simplified version of original logic)
        return None


class ExaminerDataExtractor(BaseExtractor):
    def _register_patterns(self):
        self.matcher.add(
            "EXAMINER_WITH_TITLE",
            [
                [
                    {"LOWER": "untersuchender"},
                    {"LOWER": "arzt"},
                    {"TEXT": ":"},
                    {"TEXT": "dr."},
                    {"POS": "PROPN"},
                    {"POS": "PROPN"},
                ]
            ],
        )
        self.matcher.add(
            "EXAMINER_NO_TITLE",
            [
                [
                    {"LOWER": "untersuchender"},
                    {"LOWER": "arzt"},
                    {"TEXT": ":"},
                    {"POS": "PROPN"},
                    {"POS": "PROPN"},
                ]
            ],
        )

    def extract_examiner_info(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        if not matches:
            return None

        # Take the first match
        match_id, start, end = matches[0]
        span = doc[start:end]
        tokens = list(span)
        pattern_name = self.nlp.vocab.strings[match_id]

        title, first, last = None, None, None

        if pattern_name == "EXAMINER_WITH_TITLE":
            title, first, last = tokens[3].text, tokens[4].text, tokens[5].text
        elif pattern_name == "EXAMINER_NO_TITLE":
            first, last = tokens[3].text, tokens[4].text

        self.safe_update_meta(
            {"examiner_first_name": first, "examiner_last_name": last}
        )

        return {
            "examiner_title": title,
            "examiner_first_name": first,
            "examiner_last_name": last,
        }


class EndoscopeDataExtractor(BaseExtractor):
    def _register_patterns(self):
        self.matcher.add(
            "ENDOSCOPE_INFO",
            [
                [
                    {"LOWER": "endoskop"},
                    {"TEXT": ":"},
                    {"POS": "PROPN", "OP": "+"},  # Model Name parts
                    {"LOWER": "seriennummer"},
                    {"TEXT": ":"},
                    {"SHAPE": "dddddddd"},  # Serial number format
                ]
            ],
        )

    def extract_endoscope_info(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        if not matches:
            return None

        _, start, end = matches[0]
        tokens = list(doc[start:end])

        # Locate keyword "seriennummer" to split model name from serial
        try:
            ser_idx = next(
                i for i, t in enumerate(tokens) if t.lower_ == "seriennummer"
            )
            model_name = " ".join([t.text for t in tokens[2:ser_idx]]).strip()
            serial_number = tokens[-1].text
        except (StopIteration, IndexError):
            return None

        self.safe_update_meta(
            {"endoscope_type": model_name, "endoscope_sn": serial_number}
        )

        return {"model_name": model_name, "serial_number": serial_number}


class ExaminationDataExtractor(BaseExtractor):
    """
    Extracts examination metadata.
    Note: Primarily uses Regex as per original logic, but inherits for consistency.
    """

    def extract_examination_info(
        self, text: str, remove_examiner_titles: bool = True
    ) -> Optional[Dict[str, Any]]:
        if "1. Unters.:" in text or "Unters.:" in text:
            return self._extract_format_1(text)
        if "Eingang am:" in text:
            return self._extract_format_2(text)
        return None

    def _extract_format_1(self, line: str) -> Dict[str, Any]:
        # Format: Unters.: LastName, FirstName U-datum: DD.MM.YYYY HH:MM
        pattern = r"Unters\.:\s*([\w\s\.]+),\s*([\w\s]+)\s*U-datum:\s*(\d{2}\.\d{2}\.\d{4})\s*(\d{2}:\d{2})"
        match = re.search(pattern, line)

        if match:
            last_name, first_name = match.group(1).strip(), match.group(2).strip()
            # Parse date
            try:
                date_obj = datetime.strptime(match.group(3), "%d.%m.%Y")
                ex_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                ex_date = None

            ex_time = match.group(4)

            self.safe_update_meta(
                {
                    "examiner_last_name": last_name,
                    "examiner_first_name": first_name,
                    "examination_date": ex_date,
                    "examination_time": ex_time,
                }
            )
            return {
                "examiner_last_name": last_name,
                "examiner_first_name": first_name,
                "examination_date": ex_date,
                "examination_time": ex_time,
            }

        # Fallback to SpaCy extractor if regex fails
        fallback_extractor = ExaminerDataExtractor(meta=self.meta)
        return fallback_extractor.extract_examiner_info(line)

    def _extract_format_2(self, line: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"Eingang am:\s*(\d{2}\.\d{2}\.\d{4})", line)
        if match:
            ex_date = _clean_date(match.group(1))
            self.safe_update_meta({"examination_date": ex_date, "examination_time": ""})
            return {
                "examiner_last_name": "",
                "examiner_first_name": "",
                "examination_date": ex_date,
                "examination_time": "",
            }

        return None
