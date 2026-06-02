import re
import os
from datetime import datetime
from typing import (
    Any,
    Final,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span, Token

from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.ner.determine_gender import determine_gender
from lx_anonymizer.regex_patterns import (
    DATE_8_DIGIT_RE,
    DATE_TEXT_RE,
    MULTISPACE_RE,
    REPORT_ENTRY_DATE_RE,
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta

logger = get_logger(__name__)

# --- Constants ---

TITLE_WORDS: Final[set[str]] = {
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

DATE_RE: Pattern[str] = DATE_TEXT_RE


class PatientInfo(TypedDict):
    first_name: Optional[str]
    last_name: Optional[str]
    dob: Optional[str]
    casenumber: Optional[str]
    gender: Optional[str]


class ExaminerInfo(TypedDict):
    examiner_title: Optional[str]
    examiner_first_name: Optional[str]
    examiner_last_name: Optional[str]


class EndoscopeInfo(TypedDict):
    model_name: Optional[str]
    serial_number: Optional[str]


class ExaminationInfo(TypedDict):
    examiner_last_name: Optional[str]
    examiner_first_name: Optional[str]
    examination_date: Optional[str]
    examination_time: Optional[str]


# --- Utilities ---


class SpacyModelManager:
    """
    Singleton-like manager to ensure the model is loaded only once.
    """

    _instance: Optional[Language] = None
    DEFAULT_MODEL = "de_core_news_sm"
    MODEL_ENV = "LX_ANONYMIZER_SPACY_MODEL"
    AUTO_DOWNLOAD_ENV = "LX_ANONYMIZER_SPACY_AUTO_DOWNLOAD"
    STRICT_MODEL_ENV = "LX_ANONYMIZER_SPACY_STRICT"
    _TRUE_VALUES = {"1", "true", "yes", "on"}

    @classmethod
    def configured_model_name(cls) -> str:
        return (
            os.environ.get(cls.MODEL_ENV, cls.DEFAULT_MODEL).strip()
            or cls.DEFAULT_MODEL
        )

    @classmethod
    def _env_flag_enabled(cls, env_name: str) -> bool:
        return os.environ.get(env_name, "").strip().casefold() in cls._TRUE_VALUES

    @classmethod
    def get_model(cls, model_name: Optional[str] = None) -> Language:
        if cls._instance is not None:
            return cls._instance

        model_name = model_name or cls.configured_model_name()

        try:
            logger.info(f"Loading spacy model: {model_name}")
            cls._instance = spacy.load(model_name)
        except OSError as exc:
            if cls._env_flag_enabled(cls.AUTO_DOWNLOAD_ENV):
                logger.warning(
                    "Model '%s' not found. Attempting download because %s is enabled.",
                    model_name,
                    cls.AUTO_DOWNLOAD_ENV,
                )
                cls._download_model(model_name)
                cls._instance = spacy.load(model_name)
                logger.info(f"Successfully loaded {model_name} after download.")
            elif cls._env_flag_enabled(cls.STRICT_MODEL_ENV):
                message = cls._missing_model_message(model_name)
                logger.error(message)
                raise RuntimeError(message) from exc
            else:
                cls._instance = cls._fallback_model(model_name)

        return cls._instance

    @classmethod
    def _download_model(cls, model_name: str) -> None:
        try:
            from spacy.cli.download import download as download_model

            download_model(model_name)
        except SystemExit as exc:
            raise RuntimeError(
                "spaCy model download failed with exit code "
                f"{exc.code!r}. Install '{model_name}' in the runtime environment "
                f"or disable {cls.AUTO_DOWNLOAD_ENV}."
            ) from exc

    @classmethod
    def _fallback_model(cls, model_name: str) -> Language:
        lang = cls._language_code_for_model(model_name)
        logger.warning(
            "spaCy model '%s' is not installed. Falling back to a blank '%s' "
            "pipeline; NER and POS-based extraction are degraded. Install the "
            "model or set %s=1 to allow an explicit runtime download.",
            model_name,
            lang,
            cls.AUTO_DOWNLOAD_ENV,
        )
        nlp = spacy.blank(lang)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    @staticmethod
    def _language_code_for_model(model_name: str) -> str:
        prefix = model_name.split("_", maxsplit=1)[0].strip().lower()
        if prefix.isalpha() and 2 <= len(prefix) <= 3:
            return prefix
        return "de"

    @classmethod
    def _missing_model_message(cls, model_name: str) -> str:
        return (
            f"spaCy model '{model_name}' is not installed. Install it in the "
            "runtime environment, set "
            f"{cls.AUTO_DOWNLOAD_ENV}=1 to allow an explicit runtime download, "
            f"or unset {cls.STRICT_MODEL_ENV} to use the degraded blank fallback."
        )


def _clean_date(date_str: str) -> Optional[str]:
    """
    Normalizes date strings to YYYY-MM-DD.
    Handles: dd.mm.yyyy, dd mm yyyy, ddmmyyyy (8 digits).
    """
    if not date_str:
        return None

    date_str = date_str.strip()
    # Normalize separators
    normalized = MULTISPACE_RE.sub(".", date_str)

    # 1. Try 8-digit format (DDMMYYYY) explicitly first
    if DATE_8_DIGIT_RE.fullmatch(date_str):
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

    def _register_patterns(self) -> None:
        """Subclasses should override this to add patterns to self.matcher"""
        pass

    def safe_update_meta(self, data: Mapping[str, object]) -> None:
        """Helper wrapper for meta updates."""
        self.meta.safe_update(data)


# --- Extractors ---


class PatientDataExtractor(BaseExtractor):
    def _register_patterns(self) -> None:
        # Header variations
        header_variants = r"(?i)^(pat(ient|ientin|\.?)|pationt|patbien)$"

        pat_header = [
            {"LOWER": {"REGEX": header_variants}},
            {"TEXT": ":", "OP": "?"},
        ]

        name_stop_words = sorted(
            TITLE_WORDS
            | {
                "geb",
                "geb.",
                "geboren",
                "fallnr",
                "fallnr.",
                "fallnummer",
            }
        )
        name_part = {
            "IS_ALPHA": True,
            "LOWER": {"NOT_IN": name_stop_words},
            "OP": "+",
        }
        space = {"IS_SPACE": True, "OP": "*"}

        geb_block = [
            {"LOWER": {"IN": ["geb", "geb.", "geboren"]}},
            {"TEXT": ".", "OP": "?"},
            {"LOWER": "am", "OP": "?"},
            {"TEXT": ":", "OP": "?"},
            {"TEXT": {"REGEX": DATE_RE.pattern}},
        ]

        fall_block = [
            {"LOWER": {"REGEX": r"^fall(?:nr\.?|nummer)$"}},
            {"TEXT": ".", "OP": "?"},
            {"TEXT": ":", "OP": "?"},
            {"TEXT": {"REGEX": r"[\w/-]+"}},
        ]

        # Consolidated pattern
        patient_with_dob = cast(
            list[dict[str, Any]],
            pat_header
            + [space]
            + [name_part]  # Last name (rough approx)
            + [space, {"TEXT": ",", "OP": "?"}, space]
            + [name_part]  # First name (rough approx)
            + [space]
            + [{"OP": "?"}]  # Optional token between name and DOB
            + geb_block,
        )
        patient_with_case = cast(
            list[dict[str, Any]],
            patient_with_dob
            + [space]
            + [{"OP": "?"}]
            + fall_block,
        )

        self.matcher.add("PATIENT_LINE", [patient_with_case, patient_with_dob])

    def __call__(self, text: str) -> PatientInfo:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        if not matches:
            return PatientInfo(
                first_name=self.meta.first_name,
                last_name=self.meta.last_name,
                dob=self.meta.dob.isoformat() if self.meta.dob else None,
                casenumber=self.meta.casenumber,
                gender=self.meta.gender,
            )

        # Get longest match
        _, start, end = max(matches, key=lambda m: m[2] - m[1])
        span = doc[start:end]
        tokens = list(span)

        # Extraction logic helpers
        first_name, last_name = self._extract_names(tokens)
        birthdate = self._extract_dob(span, tokens)
        case_num = self._extract_case_number(tokens)
        gender = determine_gender(first_name) if first_name else None

        self.safe_update_meta(
            {
                "first_name": first_name,
                "last_name": last_name,
                "dob": birthdate,
                "casenumber": case_num,
                "gender": gender,
            }
        )

        return PatientInfo(
            first_name=self.meta.first_name,
            last_name=self.meta.last_name,
            dob=self.meta.dob.isoformat() if self.meta.dob else None,
            casenumber=self.meta.casenumber,
            gender=self.meta.gender,
        )

    def _extract_names(
        self, tokens: Sequence[Token]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Heuristics to split names based on commas or position."""
        header_end_idx = 0
        # Find where "Patient:" ends
        patient_header_words = {
            "patient",
            "patientin",
            "pat",
            "pat.",
            "patbien",
            "pationt",
        }
        for i, token in enumerate(tokens):
            if token.lower_ in patient_header_words:
                header_end_idx = i + 1
                continue
            if header_end_idx > 0 and token.text in {":", "."}:
                header_end_idx = i + 1
                continue
            if header_end_idx > 0:
                break

        comma_idx = next((i for i, t in enumerate(tokens) if t.text == ","), None)

        # Identify stop words (geb, fallnr) to stop grabbing name tokens
        stop_indices = [
            i
            for i, t in enumerate(tokens)
            if t.lower_
            in ["geb", "geb.", "geboren", "fallnr", "fallnr.", "fallnummer"]
        ]
        end_name_idx = min(stop_indices) if stop_indices else len(tokens)

        if comma_idx is not None:
            ln_tokens = tokens[header_end_idx:comma_idx]
            fn_tokens = tokens[comma_idx + 1 : end_name_idx]
        else:
            # Fallback: assume Lastname Firstname order without comma
            full_name_tokens = tokens[header_end_idx:end_name_idx]
            if len(full_name_tokens) >= 2:
                # Without a comma the original source appears to use "LastName FirstName".
                ln_tokens = full_name_tokens[:-1]
                fn_tokens = full_name_tokens[-1:]
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

    def _extract_dob(self, span: Span, tokens: Sequence[Token]) -> Optional[str]:
        date_match = DATE_RE.search(span.text)
        if not date_match:
            return None

        # Find the specific token that matches the regex
        for token in tokens:
            if DATE_RE.fullmatch(token.text):
                return _clean_date(token.text)
        return None

    def _extract_case_number(self, tokens: Sequence[Token]) -> Optional[str]:
        # Look for explicitly labeled case number
        for i, t in enumerate(tokens):
            if t.lower_ in ["fallnr", "fallnr.", "fallnummer"]:
                # Check neighbors
                candidates = tokens[i + 1 : i + 5]
                for cand in candidates:
                    if cand.text not in {":", "."} and re.fullmatch(
                        r"[\w/-]+", cand.text
                    ):
                        return cand.text

        # Fallback: Find alphanumeric token at end of string that isn't the DOB
        # (This is a simplified version of original logic)
        return None

    @staticmethod
    def _blank() -> PatientInfo:
        return PatientInfo(
            first_name=None,
            last_name=None,
            dob=None,
            casenumber=None,
            gender=None,
        )


class ExaminerDataExtractor(BaseExtractor):
    def _register_patterns(self) -> None:
        self.matcher.add(
            "EXAMINER_WITH_TITLE",
            [
                [
                    {"LOWER": "untersuchender"},
                    {"LOWER": "arzt"},
                    {"TEXT": ":"},
                    {"LOWER": {"IN": ["dr", "dr."]}},
                    {"TEXT": ".", "OP": "?"},
                    {"IS_ALPHA": True},
                    {"IS_ALPHA": True},
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
                    {"IS_ALPHA": True},
                    {"IS_ALPHA": True},
                ]
            ],
        )

    def extract_examiner_info(self, text: str) -> Optional[ExaminerInfo]:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        if not matches:
            return None

        # Take the first match
        match_id, start, end = matches[0]
        span = doc[start:end]
        tokens = list(span)
        pattern_name = self.nlp.vocab.strings[match_id]

        title: Optional[str] = None
        first: Optional[str] = None
        last: Optional[str] = None

        content_tokens = [
            token for token in tokens[3:] if not token.is_space and token.text != "."
        ]

        if pattern_name == "EXAMINER_WITH_TITLE" and len(content_tokens) >= 3:
            title = content_tokens[0].text
            first = content_tokens[1].text
            last = content_tokens[2].text
        elif pattern_name == "EXAMINER_NO_TITLE" and len(content_tokens) >= 2:
            first = content_tokens[0].text
            last = content_tokens[1].text
        else:
            logger.debug(
                "Unexpected examiner token layout for pattern %s", pattern_name
            )
            return None

        self.safe_update_meta(
            {"examiner_first_name": first, "examiner_last_name": last}
        )

        return ExaminerInfo(
            examiner_title=title,
            examiner_first_name=first,
            examiner_last_name=last,
        )


class EndoscopeDataExtractor(BaseExtractor):
    def _register_patterns(self) -> None:
        self.matcher.add(
            "ENDOSCOPE_INFO",
            [
                [
                    {"LOWER": "endoskop"},
                    {"TEXT": ":"},
                    {
                        "TEXT": {"NOT_IN": [":"]},
                        "LOWER": {"NOT_IN": ["seriennummer"]},
                        "OP": "+",
                    },
                    {"LOWER": "seriennummer"},
                    {"TEXT": ":"},
                    {"SHAPE": "dddddddd"},  # Serial number format
                ]
            ],
        )

    def extract_endoscope_info(self, text: str) -> Optional[EndoscopeInfo]:
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

        return EndoscopeInfo(model_name=model_name, serial_number=serial_number)


class ExaminationDataExtractor(BaseExtractor):
    """
    Extracts examination metadata.
    Note: Primarily uses Regex as per original logic, but inherits for consistency.
    """

    def extract_examination_info(
        self, text: str, remove_examiner_titles: bool = True
    ) -> Optional[ExaminationInfo]:
        if "1. Unters.:" in text or "Unters.:" in text:
            return self._extract_format_1(
                text, remove_examiner_titles=remove_examiner_titles
            )
        if "Eingang am:" in text:
            return self._extract_format_2(text)
        return None

    def _extract_format_1(
        self, line: str, remove_examiner_titles: bool = True
    ) -> Optional[ExaminationInfo]:
        # Format: Unters.: LastName, FirstName U-datum: DD.MM.YYYY HH:MM
        pattern = r"Unters\.:\s*([\w\s\.]+),\s*([\w\s]+)\s*U-datum:\s*(\d{2}\.\d{2}\.\d{4})\s*(\d{2}:\d{2})"
        match = re.search(pattern, line)

        if match:
            last_name, first_name = match.group(1).strip(), match.group(2).strip()
            if remove_examiner_titles:
                last_name = self._remove_title_words(last_name)
                first_name = self._remove_title_words(first_name)
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
            return ExaminationInfo(
                examiner_last_name=last_name,
                examiner_first_name=first_name,
                examination_date=ex_date,
                examination_time=ex_time,
            )

        # Fallback to SpaCy extractor if regex fails
        fallback_extractor = ExaminerDataExtractor(meta=self.meta)
        examiner_info = fallback_extractor.extract_examiner_info(line)
        if examiner_info is None:
            return None
        first_name = examiner_info["examiner_first_name"]
        last_name = examiner_info["examiner_last_name"]
        if remove_examiner_titles:
            if first_name:
                first_name = self._remove_title_words(first_name)
            if last_name:
                last_name = self._remove_title_words(last_name)
        return ExaminationInfo(
            examiner_last_name=last_name,
            examiner_first_name=first_name,
            examination_date=(
                self.meta.examination_date.isoformat()
                if self.meta.examination_date
                else None
            ),
            examination_time=(
                self.meta.examination_time.isoformat()
                if self.meta.examination_time
                else None
            ),
        )

    def _extract_format_2(self, line: str) -> Optional[ExaminationInfo]:
        match = REPORT_ENTRY_DATE_RE.search(line)
        if match:
            ex_date = _clean_date(match.group(1))
            self.safe_update_meta({"examination_date": ex_date, "examination_time": ""})
            return ExaminationInfo(
                examiner_last_name="",
                examiner_first_name="",
                examination_date=ex_date,
                examination_time="",
            )

        return None

    @staticmethod
    def _remove_title_words(name: str) -> str:
        parts = [part for part in name.split() if part.casefold() not in TITLE_WORDS]
        return " ".join(parts).strip()
