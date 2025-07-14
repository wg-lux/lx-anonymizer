import re
from django.template.defaultfilters import first, last
import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
from datetime import datetime
import warnings
from .determine_gender import determine_gender
import re, json, spacy
from spacy.language import Language
from spacy.tokens import Doc
from .spacy_extractor import _clean_date

# Compile heavy regexes once at module level
DATE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")
NUMBER_RE = re.compile(r"^\d+$")

class PatientDataExtractorLg:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_lg")
        # Initialize ruler here and add it to the pipeline
        # overwrite_ents=True allows the ruler to overwrite existing entities if needed
        if "entity_ruler" not in self.nlp.pipe_names:
            self.ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, before="ner")
        else:
            self.ruler = self.nlp.get_pipe("entity_ruler")
        
        # FIX: Actually register the regex patterns
        self.ruler.add_patterns(self._build_regex_patterns())
        
        self.matcher = Matcher(self.nlp.vocab) # Initialize matcher here
        self.setup_matcher()
        Doc.set_extension("meta", default={}, force=True)
        
        # Initialize examiner ruler once
        self._examiner_ruler = None
    
    _FIELDS = {
    "patient_first_name": r"patient_first_name",
    "patient_last_name":  r"patient_last_name",
    "patient_dob":        r"(?:patient_)?(?:dob|geb(?:urtsdatum)?)",
    "casenumber":         r"(?:case)?number",
    "patient_gender":     r"(?:patient_)?gender",
    }
    # Improved VALUE pattern to handle UTF-8 names with accents
    _VALUE = r'["\']?(?P<val>[\w\.\-\/ äöüßÄÖÜéèêç]+?)["\']?$'
    
    clean_date = _clean_date

    def _build_regex_patterns(self):
        patts = []
        for key, lbl in self._FIELDS.items():
            # (?m) → multiline; ^\s*-? optional bullet or list dash
            pattern = rf"(?m)^\s*-?\s*{lbl}\s*[:=]\s*{self._VALUE}"
            patts.append({"label": key, "pattern": pattern})
        return patts


    def regex_extract_llm_meta(self, text: str) -> dict[str, str | None]:
        """
        Parse an LLM answer *that is NOT valid JSON* and recover
        first name / last name / dob / casenumber / gender.

        Returns a dict with the same keys as PatientMeta – values may be None
        """
        doc = self.nlp(text)
        meta = {k: None for k in self._FIELDS}      # initialise blanks

        for ent in doc.ents:                   # RegexRuler creates ents
            key = ent.label_
            # drop the field name, keep only captured value (last group)
            m = re.search(self._VALUE, ent.text)
            if not m:
                continue
            val = m.group("val").strip()
            # basic normalisation
            if key == "patient_dob":
                # try to normalise dd.mm.yyyy → yyyy-mm-dd (reuse your _clean_date)
                val = self.clean_date(val) or val
            meta[key] = val

        # sanity: if either name field still None but the other has value "Unknown"
        for k in ("patient_first_name", "patient_last_name"):
            if meta[k] and meta[k].lower() in ("unknown", "unbekannt"):
                meta[k] = None

        return meta



    def setup_matcher(self):
        """Setup the SpaCy Matcher and Ruler with robust patient information patterns"""
        # Define pattern components using token attributes
        pat_header = [
            # Matches "Pat.", "Patient", "Patientin" at the start of the token, case-insensitive
            {"LOWER": {"REGEX": r"^pat(?:ient|ientin|\.?)$"}},
            # Optional colon directly after
            {"TEXT": ":", "OP": "?"}
        ]
        name_block = [
            # One or more capitalized words (likely last name)
            {"IS_TITLE": True, "OP": "+"},
            # Comma separator
            {"TEXT": ","},
            # One or more capitalized words (likely first name)
            {"IS_TITLE": True, "OP": "+"}
        ]
        geb_block = [
            # Matches "geb", "geb.", "geboren"
            {"LOWER": {"IN": ["geb", "geb.", "geboren"]}},
            # Optional "am"
            {"LOWER": "am", "OP": "?"},
             # Optional colon
            {"TEXT": ":", "OP": "?"},
            # Date in dd.dd.dddd format
            {"TEXT": {"REGEX": r"^\d{1,2}\.\d{1,2}\.\d{4}$"}}
        ]
        fall_block = [
            # Matches "Fallnr", "Fallnr.", "Fallnummer"
            {"LOWER": {"REGEX": r"^fall(?:nr\.?|nummer)$"}},
            # Optional colon
            {"TEXT": ":", "OP": "?"},
            # One or more digits
            {"TEXT": {"REGEX": r"^\d+$"}}
        ]

        # FIX: spacer should be a list containing the dictionary, not just a dictionary
        spacer = [{"OP": "*", "IS_SPACE": True}]  # Allows zero or more spaces/newlines
        
        # Combine components into the full pattern
        # Structure: Header + Name + Optional(Geb) + Optional(Fall)
        pattern = pat_header + spacer + name_block + spacer + \
                  [{"OP": "?"}] + geb_block + spacer + \
                  [{"OP": "?"}] + fall_block

        # Add the combined pattern to Matcher and Ruler
        self.matcher.add("PATIENT_LINE", [pattern])
        self.ruler.add_patterns([{"label": "PATIENT_INFO", "pattern": pattern}])

        # Debugging: Check if patterns are loaded
        if not self.matcher:
             raise ValueError("Matcher patterns were not added.")
        if len(self.ruler.patterns) == 0:
            warnings.warn("No patterns were added to the EntityRuler in setup_matcher.")


    def patient_data_extractor(self, text):
        # Simple alias without deprecation spam
        return self.extract_patient_info(text)

    def extract_patient_info(self, text):
        """
        Extract patient information using SpaCy's matcher based on token patterns.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        # Find the longest match if multiple overlap
        best_match = None
        longest_len = 0
        for match_id, start, end in matches:
             if end - start > longest_len:
                 longest_len = end - start
                 best_match = (match_id, start, end)

        if best_match:
            match_id, start, end = best_match
            span = doc[start:end]
            # Initialize default values - FIX: Use None instead of "NOT FOUND"
            first_name, last_name, birthdate, casenumber = None, None, None, None

            # --- Extract based on token properties within the span ---
            comma_indices = [i for i, token in enumerate(span) if token.text == ","]
            geb_indices = [i for i, token in enumerate(span) if token.lower_ in ["geb", "geb.", "geboren"]]
            fall_indices = [i for i, token in enumerate(span) if token.lower_ in ["fallnr", "fallnr.", "fallnummer"]]
            date_indices = [i for i, token in enumerate(span) if DATE_RE.match(token.text)]
            case_num_indices = [i for i, token in enumerate(span) if NUMBER_RE.match(token.text) and i > 0 and span[i-1].lower_ in [":", "fallnr", "fallnr.", "fallnummer"]]

            # Extract Name (assuming structure: Header, Lastname, Comma, Firstname)
            if comma_indices:
                comma_idx = comma_indices[0]
                # Find the start of the name block (after header and optional colon)
                name_start_idx = 1
                if span[1].text == ':':
                    name_start_idx = 2
                # Last name is between header and comma
                last_name_tokens = [t.text for t in span[name_start_idx:comma_idx] if t.is_title]
                if last_name_tokens:
                    last_name = " ".join(last_name_tokens)

                # First name is after comma until the next block (geb or fall) or end
                name_end_idx = min(geb_indices + fall_indices + [len(span)])
                first_name_tokens = [t.text for t in span[comma_idx + 1 : name_end_idx] if t.is_title]
                if first_name_tokens:
                    first_name = " ".join(first_name_tokens)

            # Extract Birthdate
            if date_indices:
                # Find the date that likely follows a 'geb' keyword
                date_idx = -1
                if geb_indices:
                    geb_idx = geb_indices[0]
                    possible_dates = [d_idx for d_idx in date_indices if d_idx > geb_idx]
                    if possible_dates:
                        date_idx = min(possible_dates) # Take the first date after 'geb'
                elif date_indices: # If no 'geb' but date exists
                     date_idx = date_indices[0]

                if date_idx != -1:
                    birthdate_str = span[date_idx].text
                    try:
                        birthdate = datetime.strptime(birthdate_str, "%d.%m.%Y").strftime("%Y-%m-%d")
                    except ValueError:
                        birthdate = None # Mark as None instead of "INVALID DATE FORMAT"

            # Extract Casenumber
            if case_num_indices:
                # Find the number that likely follows a 'fall' keyword
                num_idx = -1
                if fall_indices:
                    fall_idx = fall_indices[0]
                    possible_nums = [c_idx for c_idx in case_num_indices if c_idx > fall_idx]
                    if possible_nums:
                        num_idx = min(possible_nums) # Take the first number after 'fall'
                elif case_num_indices: # If no 'fall' but number exists (less reliable)
                    num_idx = case_num_indices[-1] # Take the last number found

                if num_idx != -1:
                    casenumber = span[num_idx].text

            # Determine Gender
            gender = determine_gender(first_name) if first_name else None

            return {
                "patient_first_name": first_name,
                "patient_last_name": last_name,
                "patient_dob": birthdate,
                "casenumber": casenumber,
                "patient_gender": gender
            }

        # Fallback if no match found by the matcher - FIX: Return None values
        return {
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender": None
        }

    def _extract_using_entity_ruler(self, text):
        """
        Helper method potentially using entity ruler results.
        NOTE: The string patterns previously here were problematic.
        This method might need redesign or removal depending on usage.
        It currently relies on the ruler patterns set in setup_matcher.
        """
        doc = self.nlp(text)
        patient_info_ents = [ent for ent in doc.ents if ent.label_ == "PATIENT_INFO"]

        if patient_info_ents:
            # Process the found entities if needed, e.g., extract details
            # This logic would be similar to the extraction in extract_patient_info
            # For now, just return a placeholder indicating an entity was found
            return {"status": "PATIENT_INFO entity found by ruler"}
        else:
            # Fallback or further processing if no PATIENT_INFO entity found
            # Example: Look for general PER entities (less reliable)
            person_ents = [ent.text for ent in doc.ents if ent.label_ == "PER"]
            if person_ents:
                 return {"status": "PER entities found", "persons": person_ents}

        return {
            'status': "No relevant entities found by ruler",
            'patient_first_name': "Unknown",
            'patient_last_name': "Unknown",
            'birthdate': None,
            'casenumber': None,
            'patient_gender': "Unknown"
        }

    def examiner_data_extractor(self, text):
        # FIX: Reuse self.nlp instead of loading spacy.load("de_core_news_lg") every time
        # Initialize examiner ruler once if not already done
        if not self._examiner_ruler:
            patterns = [
                {"label": "PER", "pattern": r"Dr\.\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.med\.\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.med\.\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.med\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.med\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
                {"label": "PER", "pattern": r"Dr\.med\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},  
            ]
            self._examiner_ruler = self.nlp.add_pipe("entity_ruler", name="examiner_ruler", last=True)
            self._examiner_ruler.add_patterns(patterns)
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        return None


