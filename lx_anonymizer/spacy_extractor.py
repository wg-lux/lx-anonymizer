import spacy
from spacy.matcher import Matcher
from datetime import datetime
import re
from .determine_gender import determine_gender
from .custom_logger import get_logger
from .sensitive_meta_interface import SensitiveMeta
from typing import Optional, Dict, Any  
# Import spacy's download function
import spacy.cli

logger = get_logger(__name__)
# import spacy language model
from spacy.language import Language


# Define a set of title words to ignore (lowercase)
TITLE_WORDS = {
    "herr", "herrn", "frau", "f", "fru", "fruis", "fruharuka",  # 'fruis', 'fruharuka' likely OCR errors
    "dr.", "dr", "doctor", "prof.", "prof", "professor", "ing.", "ing",
    "señor", "señorita", "monsieur", "mr.", "mrs.", "sir", "mag.", "baron"
}

def load_spacy_model(model_name:str="de_core_news_lg") -> "Language":
    try:
        nlp_model = spacy.load(model_name)
        logger.info(f"Successfully loaded spacy model: {model_name}")
    except OSError as e:
        logger.warning(f"Spacy model '{model_name}' not found. Attempting download...")
        try:
            # Use spacy's built-in download command
            spacy.cli.download(model_name)
            logger.info(f"Successfully downloaded spacy model: {model_name}")
            # Try loading again after download
            nlp_model = spacy.load(model_name)
            logger.info(f"Successfully loaded spacy model after download: {model_name}")
        except Exception as download_exc:
            logger.error(f"Failed to download or load spacy model '{model_name}' after attempt: {download_exc}")
            raise download_exc # Re-raise the exception if download/load fails

    return nlp_model

# Updated DATE_RE to also accept 8 digits without separators (DDMMYYYY)
DATE_RE = re.compile(r"(\d{1,2}[.\s]?\d{1,2}[.\s]?\d{2,4})|(\d{8})")

def _clean_date(date_str: str) -> str | None:
    """Converts various date formats including dd.mm.yyyy, dd mm yyyy, ddmmyyyy to YYYY-MM-DD."""
    date_str = date_str.strip()
    normalized_date_str = re.sub(r'\s+', '.', date_str) # Normalize spaces to dots
    normalized_date_str = re.sub(r'[^\d.]', '', normalized_date_str) # Keep only digits and dots

    formats_to_try = [
        "%d.%m.%Y",  # 01.12.2023
        "%d.%m.%y",  # 01.12.23
        "%d%m%Y",    # 01122023 (8 digits) - Added
    ]
    dt_obj = None

    # Handle the 8-digit case first if it matches
    if re.fullmatch(r"\d{8}", date_str):
        try:
            # Ensure day and month are valid before parsing
            day = int(date_str[0:2])
            month = int(date_str[2:4])
            year = int(date_str[4:8])
            if 1 <= day <= 31 and 1 <= month <= 12:
                 dt_obj = datetime.strptime(date_str, "%d%m%Y")
            else:
                 logger.warning(f"Invalid day/month in 8-digit date: {date_str}")
        except ValueError:
            pass # Will be caught later if no format matches

    # Try other formats if 8-digit parsing failed or didn't apply
    if not dt_obj:
        for fmt in formats_to_try:
             # Skip 8-digit format here if already tried or doesn't match input structure
             if fmt == "%d%m%Y" and not re.fullmatch(r"\d{8}", normalized_date_str):
                 continue
             # Use normalized string for dot-separated formats
             current_str_to_try = date_str if fmt == "%d%m%Y" else normalized_date_str
             try:
                 dt_obj = datetime.strptime(current_str_to_try, fmt)
                 # Handle two-digit year ambiguity
                 if dt_obj.year < 100:
                     if dt_obj.year >= 69: # Assuming 69-99 are 19xx
                         dt_obj = dt_obj.replace(year=dt_obj.year + 1900)
                     else: # Assuming 00-68 are 20xx
                         dt_obj = dt_obj.replace(year=dt_obj.year + 2000)
                 break # Stop if parsing is successful
             except ValueError:
                 continue # Try next format

    if dt_obj:
        return dt_obj.strftime("%Y-%m-%d")
    else:
        logger.warning(f"Invalid or unparseable date format encountered: {date_str}")
        return None

class PatientDataExtractor:
    """
    Rule-based header line extractor for German medical reports.
    Uses SpaCy Matcher with token-based patterns.
    Loads the SpaCy model once and reuses it.
    """

    _nlp: Language | None = None
    _matcher: Matcher | None = None
    _rules_built = False

    def __init__(self, meta: Optional[SensitiveMeta] = None) -> None:
        if PatientDataExtractor._nlp is None:
            PatientDataExtractor._nlp = load_spacy_model("de_core_news_lg")

        if not PatientDataExtractor._rules_built:
            PatientDataExtractor._matcher = Matcher(PatientDataExtractor._nlp.vocab)
            self._build_rules()
            PatientDataExtractor._rules_built = True

        self._nlp = PatientDataExtractor._nlp
        self._matcher = PatientDataExtractor._matcher
        self.meta: SensitiveMeta = meta or SensitiveMeta()

    def _build_rules(self) -> None:
        """Builds the token-based patterns including OCR variants."""
        assert PatientDataExtractor._matcher is not None, "PatientDataExtractor._matcher should be initialized before building rules" # Added assertion
        HEADER_VARIANTS = [
            r"^pat(?:ient|ientin|\.?)$",
            r"^pationt$",
            r"^patbien$",
        ]
        pat_header = [
            {"LOWER": {"REGEX": "(" + "|".join(HEADER_VARIANTS) + ")"}},
            {"TEXT": ":", "OP": "?"}
        ]
        name_tokens = [
             {"OP": "+", "IS_TITLE": True}
        ]
        comma_sep = {"TEXT": ","}

        geb_block = [
            {"LOWER": {"IN": ["geb", "geb.", "geboren"]}},
            {"LOWER": "am", "OP": "?"},
            {"TEXT": ":", "OP": "?"},
            {"TEXT": {"REGEX": DATE_RE.pattern}}
        ]
        fall_block = [
            {"LOWER": {"REGEX": r"^fall(?:nr\.?|nummer)$"}},
            {"TEXT": ":",  "OP": "?"},
            {"TEXT": {"REGEX": r"[\w/-]+"}}
        ]
        space = {"IS_SPACE": True, "OP": "*"}

        full_pattern = pat_header + [space] + \
                       name_tokens + [comma_sep] + [space] + \
                       name_tokens + [space] + \
                       [{"OP": "?"}] + geb_block + [space] + \
                       [{"OP": "?"}] + fall_block

        name_part = {"POS": {"IN": ["PROPN", "NOUN"]}, "IS_TITLE": True, "OP": "+"}
        simpler_pattern = pat_header + [space] + \
                          [name_part] + [space, {"TEXT": ",", "OP": "?"}, space] + \
                          [name_part] + [space] + \
                          [{"OP": "?"}] + geb_block + [space] + \
                          [{"OP": "?"}] + fall_block

        PatientDataExtractor._matcher.add("PATIENT_LINE", [simpler_pattern])
        logger.debug("SpaCy rules for PatientDataExtractor built (Matcher only).")

    def __call__(self, text: str) -> dict[str, str | None]:
        if not self._nlp or not self._matcher:
            logger.error("SpaCy NLP model or Matcher not initialized.")
            # still return a dict, and do not mutate meta here
            return self._blank()

        doc = self._nlp(text)
        matches = self._matcher(doc)

        if not matches:
            return self.meta.to_dict()  # keep existing values if any

        match_id, start, end = max(matches, key=lambda m: m[2] - m[1])
        span = doc[start:end]
        logger.debug(f"Matched span: '{span.text}'")

        tokens = list(span)
        first_name, last_name, birthdate, case_num = None, None, None, None

        comma_indices = [i for i, t in enumerate(tokens) if t.text == ","]
        header_end_idx = 0
        for i, token in enumerate(tokens):
            if token.lower_ in ["patient", "patientin", "pat.", "pationt", "patbien", ":"]:
                 header_end_idx = i + 1
            else:
                 if tokens[header_end_idx-1].text != ':':
                     break

        if comma_indices:
            comma_idx = comma_indices[0]
            last_name_candidates = [t for t in tokens[header_end_idx:comma_idx] if t.is_alpha or t.like_num]
            geb_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["geb", "geb.", "geboren"]]
            fall_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["fallnr", "fallnr.", "fallnummer"]]
            non_name_start_idx = min(geb_indices + fall_indices + [len(tokens)])

            first_name_candidates = [t for t in tokens[comma_idx + 1 : non_name_start_idx] if t.is_alpha or t.like_num]

            last_name_filtered = [t.text for t in last_name_candidates if t.lower_ not in TITLE_WORDS]
            first_name_filtered = [t.text for t in first_name_candidates if t.lower_ not in TITLE_WORDS]

            if last_name_filtered:
                last_name = " ".join(last_name_filtered)
            if first_name_filtered:
                first_name = " ".join(first_name_filtered)
        else:
            logger.warning(f"No comma found in matched span: '{span.text}'. Name splitting might be inaccurate.")
            geb_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["geb", "geb.", "geboren"]]
            fall_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["fallnr", "fallnr.", "fallnummer"]]
            non_name_start_idx = min(geb_indices + fall_indices + [len(tokens)])
            name_candidates = [t for t in tokens[header_end_idx:non_name_start_idx] if t.is_alpha or t.like_num]
            name_filtered = [t.text for t in name_candidates if t.lower_ not in TITLE_WORDS]
            if len(name_filtered) >= 2:
                 last_name = " ".join(name_filtered)
            elif len(name_filtered) == 1:
                 last_name = name_filtered[0]

        date_match = DATE_RE.search(span.text)
        if date_match:
            raw_date_str = date_match.group(0)
            date_token = next((t for t in tokens if t.idx >= span.start_char + date_match.start() and t.idx < span.start_char + date_match.end() and DATE_RE.fullmatch(t.text)), None)
            if date_token:
                 prev_token_idx = date_token.i - span.start -1
                 if prev_token_idx >= 0:
                     prev_token = tokens[prev_token_idx]
                     if prev_token.lower_ in ["geb", "geb.", "geboren", "am", ":"]:
                         birthdate_cleaned = _clean_date(date_token.text)
                         birthdate = birthdate_cleaned if birthdate_cleaned is not None else None
                     else:
                          logger.debug(f"Date '{date_token.text}' found but not preceded by expected keyword.")
                 else:
                      birthdate_cleaned = _clean_date(date_token.text)
                      birthdate = birthdate_cleaned if birthdate_cleaned is not None else None
            else:
                 logger.debug(f"Regex matched date '{raw_date_str}' but no corresponding token found or context mismatch.")

        case_token = None
        fall_keyword_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["fallnr", "fallnr.", "fallnummer"]]
        if fall_keyword_indices:
            keyword_idx = fall_keyword_indices[0]
            if keyword_idx + 1 < len(tokens):
                next_token = tokens[keyword_idx + 1]
                if next_token.text == ":" and keyword_idx + 2 < len(tokens):
                    potential_case_token = tokens[keyword_idx + 2]
                    if re.fullmatch(r"[\w/-]+", potential_case_token.text):
                        case_token = potential_case_token
                elif re.fullmatch(r"[\w/-]+", next_token.text):
                     case_token = next_token

        if case_token:
            case_num = case_token.text
        else:
             geb_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["geb", "geb.", "geboren"]]
             fall_indices = [i for i, t in enumerate(tokens) if t.lower_ in ["fallnr", "fallnr.", "fallnummer"]]
             non_name_start_idx = min(geb_indices + fall_indices + [len(tokens)])
             potential_case_tokens = [t for i, t in enumerate(tokens) if i >= non_name_start_idx and re.fullmatch(r"[\w/-]+", t.text) and t.text != birthdate]
             if potential_case_tokens:
                  case_num = potential_case_tokens[0].text

        gender = determine_gender(first_name) if first_name else None

        self.meta.safe_update({
            "patient_first_name": first_name,
            "patient_last_name":  last_name,
            "patient_dob":        birthdate,   # ISO string or None is fine
            "casenumber":         case_num,
            "patient_gender_name": gender
        })

        # Return the living snapshot as dict (backward compatible)
        return self.meta.to_dict()

    @staticmethod
    def _blank() -> dict[str, str | None]:
        return {
            "patient_first_name": None,
            "patient_last_name":  None,
            "patient_dob":        None,
            "casenumber":         None,
            "patient_gender_name": None
        }

class ExaminerDataExtractor:
    _nlp: Language | None = None  # shared model

    def __init__(self, meta: Optional[SensitiveMeta] = None):
        if ExaminerDataExtractor._nlp is None:
            ExaminerDataExtractor._nlp = load_spacy_model("de_core_news_lg")
        self.nlp = ExaminerDataExtractor._nlp
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        self.meta: SensitiveMeta = meta or SensitiveMeta()

    def _setup_patterns(self):
        pattern1 = [
            {"LOWER": "untersuchender"},
            {"LOWER": "arzt"},
            {"TEXT": ":"},
            {"TEXT": "dr."},
            {"POS": "PROPN"},
            {"POS": "PROPN"}
        ]

        pattern2 = [
            {"LOWER": "untersuchender"},
            {"LOWER": "arzt"},
            {"TEXT": ":"},
            {"POS": "PROPN"},
            {"POS": "PROPN"}
        ]

        self.matcher.add("EXAMINER_INFO_1", [pattern1])
        self.matcher.add("EXAMINER_INFO_2", [pattern2])

    def extract_examiner_info(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]

            title, first_name, last_name = None, None, None
            tokens = list(span)

            if pattern_name == "EXAMINER_INFO_1":
                title = tokens[3].text
                first_name = tokens[4].text
                last_name = tokens[5].text

            elif pattern_name == "EXAMINER_INFO_2":
                first_name = tokens[3].text
                last_name = tokens[4].text

            self.meta.safe_update({
                "examiner_first_name": first_name,
                "examiner_last_name":  last_name
            })

            # keep returning a small dict (existing behavior)
            return {
                "examiner_title": title,
                "examiner_first_name": first_name,
                "examiner_last_name": last_name
            }

        return None

class EndoscopeDataExtractor:
    _nlp: Language | None = None

    def __init__(self, meta: Optional[SensitiveMeta] = None):
        if EndoscopeDataExtractor._nlp is None:
            EndoscopeDataExtractor._nlp = load_spacy_model("de_core_news_lg")
        self.nlp = EndoscopeDataExtractor._nlp
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        self.meta: SensitiveMeta = meta or SensitiveMeta()

    def _setup_patterns(self):
        pattern1 = [
            {"LOWER": "endoskop"},
            {"TEXT": ":"},
            {"POS": "PROPN", "OP": "+"},
            {"LOWER": "seriennummer"},
            {"TEXT": ":"},
            {"SHAPE": "dddddddd"}
        ]

        self.matcher.add("ENDOSCOPE_INFO_1", [pattern1])

    def extract_endoscope_info(self, text) -> Optional[Dict[str, Optional[str]]]:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]

            model_name, serial_number = None, None
            tokens = list(span)

            if pattern_name == "ENDOSCOPE_INFO_1":
                # model name is tokens[2:tokens.index(tokens[3])] in your code,
                # but tokens[3] *is* {"LOWER": "seriennummer"}, so safer:
                try:
                    ser_idx = next(i for i, t in enumerate(tokens) if t.lower_ == "seriennummer")
                except StopIteration:
                    ser_idx = 3  # fallback
                model_name = " ".join([t.text for t in tokens[2:ser_idx]]).strip() or None
                serial_number = tokens[-1].text if tokens else None

            # write via SensitiveMeta (mapped fields)
            self.meta.safe_update({
                "endoscope_type": model_name,
                "endoscope_sn":   serial_number
            })

            return {
                "model_name": model_name,
                "serial_number": serial_number
            }

        return None
class ExaminationDataExtractor:
    _nlp: Language | None = None

    def __init__(self, meta: Optional[SensitiveMeta] = None):
        if ExaminationDataExtractor._nlp is None:
            ExaminationDataExtractor._nlp = load_spacy_model("de_core_news_lg")
        self.nlp = ExaminationDataExtractor._nlp
        self.meta: SensitiveMeta = meta or SensitiveMeta()

    def extract_examination_info(self, text, remove_examiner_titles=True):
        if "1. Unters.:" in text or "Unters.:" in text:
            return self._extract_meta_format_1(text, remove_examiner_titles)
        if "Eingang am:" in text:
            return self._extract_meta_format_2(text, remove_examiner_titles)
        return None

    def _extract_meta_format_1(self, line, remove_examiner_titles=True):
        if remove_examiner_titles:
            pass
        pattern = r"Unters\.: ([\w\s\.]+), ([\w\s]+)\s*U-datum:\s*(\d{2}\.\d{2}\.\d{4}) (\d{2}:\d{2})"
        match = re.search(pattern, line)
        if match:
            examiner_last_name = match.group(1).strip()
            examiner_first_name = match.group(2).strip()
            examination_date = datetime.strptime(match.group(3), '%d.%m.%Y').strftime('%Y-%m-%d')
            examination_time = match.group(4)

            # write via SensitiveMeta
            self.meta.safe_update({
                "examiner_last_name":  examiner_last_name,
                "examiner_first_name": examiner_first_name,
                "examination_date":    examination_date,
                "examination_time":    examination_time
            })

            return {
                'examiner_last_name':  examiner_last_name,
                'examiner_first_name': examiner_first_name,
                'examination_date':    examination_date,
                'examination_time':    examination_time
            }
        else:
            data = ExaminerDataExtractor(meta=self.meta).extract_examiner_info(line)
            return data

    def _extract_meta_format_2(self, line, remove_examiner_titles=True):
        if remove_examiner_titles:
            pass
        pattern = r"Eingang am:\s*(\d{2}\.\d{2}\.\d{4})"
        match = re.search(pattern, line)
        if match:
            examination_date = datetime.strptime(match.group(1), '%d.%m.%Y').strftime('%Y-%m-%d')
            # write via SensitiveMeta
            self.meta.safe_update({
                'examination_date': examination_date,
                'examination_time': ""
            })
            return {
                'examiner_last_name': "",
                'examiner_first_name': "",
                'examination_date': examination_date,
                'examination_time': ""
            }
        return None
