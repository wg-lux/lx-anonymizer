import spacy
from spacy.matcher import Matcher
from datetime import datetime
import re
from .determine_gender import determine_gender
from .custom_logger import get_logger
import subprocess
from .spacy_regex import PatientDataExtractorLg

logger = get_logger(__name__)
# import spacy language model
from spacy.language import Language

def load_spacy_model(model_name:str="de_core_news_lg") -> "Language":
    try:
        nlp_model = spacy.load(model_name)

    except OSError as e:
        subprocess.run(["uv", "run", "python", "-m", "spacy", "download", model_name], check=True)
        logger.debug(f"Downloaded spacy model: {model_name} downloading after {e}")
        nlp_model = spacy.load(model_name)
    return nlp_model

class PatientDataExtractor:


    def __init__(self):
        self.logger = get_logger(__name__)

        self.nlp = load_spacy_model("de_core_news_lg")
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        """Define patterns for extracting patient information."""
        # Pattern 1: Patient: Nachname ,Vorname geb. DD.MM.YYYY Fallnummer: NNNNNNNN
        pattern1 = [
            {"LOWER": "patient"},
            {"LOWER": ":"},
            {"POS": "PROPN", "OP": "+"},
            {"TEXT": ","},
            {"POS": "PROPN", "OP": "+"},
            {"LOWER": "geb"},
            {"TEXT": "."},
            {"SHAPE": "dd.dd.dddd", "OP": "?"},
            {"LOWER": "fallnummer"},
            {"TEXT": ":"},
            {"SHAPE": "dddddddd"}
        ]

        # Pattern 2: Patient: Nachname, Vorname geboren am: DD.MM.YYYY
        pattern2 = [
            {"LOWER": "patient"},
            {"LOWER": ":"},
            {"POS": "PROPN", "OP": "+"},
            {"TEXT": ","},
            {"POS": "PROPN", "OP": "+"},
            {"LOWER": "geboren"},
            {"LOWER": "am"},
            {"TEXT": ":"},
            {"SHAPE": "dd.dd.dddd", "OP": "?"}
        ]

        self.matcher.add("PATIENT_INFO_1", [pattern1])
        self.matcher.add("PATIENT_INFO_2", [pattern2])

    def extract_patient_info(self, text):
        """
        Extract patient information from the given text.

        Parameters:
        - text (str): Input text containing patient information.

        Returns:
        - dict: Extracted patient information or None if no match is found.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]

            # Extract fields based on the matched pattern
            last_name, first_name, birthdate, casenumber = None, None, None, None
            tokens = list(span)

            if pattern_name == "PATIENT_INFO_1":
                last_name = " ".join([t.text for t in tokens[2:tokens.index(tokens[3])]])
                first_name = " ".join([t.text for t in tokens[tokens.index(tokens[3]) + 1:tokens.index(tokens[5])]])
                if tokens[-3].shape_ == "dd.dd.dddd":
                    birthdate = tokens[-3].text
                casenumber = tokens[-1].text

            elif pattern_name == "PATIENT_INFO_2":
                last_name = " ".join([t.text for t in tokens[2:tokens.index(tokens[3])]])
                first_name = " ".join([t.text for t in tokens[tokens.index(tokens[3]) + 1:tokens.index(tokens[5])]])
                if tokens[-1].shape_ == "dd.dd.dddd":
                    birthdate = tokens[-1].text

            # Format birthdate
            if birthdate:
                try:
                    birthdate = datetime.strptime(birthdate, "%d.%m.%Y").strftime("%Y-%m-%d")
                except ValueError:
                    birthdate = None
            
            gender = determine_gender(first_name)
            
            # Ensure first_name and last_name are never None
            if first_name and last_name:
                return {
                    "patient_first_name": first_name,
                    "patient_last_name": last_name,
                    "patient_dob": birthdate,
                    "casenumber": casenumber,
                    'patient_gender': gender
                }
            elif first_name and not last_name:
                return {
                    "patient_first_name": first_name,
                    "patient_last_name": "UNBEKANNT",
                    "patient_dob": birthdate,
                    "casenumber": casenumber,
                    'patient_gender': gender
                }
            elif last_name and not first_name:
                return {
                    "patient_first_name": "UNBEKANNT", 
                    "patient_last_name": last_name,
                    "patient_dob": birthdate,
                    "casenumber": casenumber,
                    'patient_gender': "Unbekannt"
                }
            else:
                # If nothing is properly extracted, try the alternative extractor
                pe = PatientDataExtractorLg()
                data = pe.extract_patient_info(text)
                if data and data.get('patient_first_name') != "NOT FOUND":
                    return data
                
                # Absolute fallback values  
                return {
                    "patient_first_name": "UNBEKANNT",
                    "patient_last_name": "UNBEKANNT",
                    "patient_dob": None,
                    "casenumber": None,
                    'patient_gender': "Unbekannt"
                }

        # If no match is found
        return {
            "patient_first_name": "UNBEKANNT", 
            "patient_last_name": "UNBEKANNT",
            "patient_dob": None,
            "casenumber": None,
            "patient_gender": "Unbekannt"
        }

class ExaminerDataExtractor:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_lg")
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        """Define patterns for extracting examiner information."""
        # Pattern 1: Untersuchender Arzt: Dr. Vorname Nachname
        pattern1 = [
            {"LOWER": "untersuchender"},
            {"LOWER": "arzt"},
            {"TEXT": ":"},
            {"TEXT": "dr."},
            {"POS": "PROPN"},
            {"POS": "PROPN"}
        ]

        # Pattern 2: Untersuchender Arzt: Vorname Nachname
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
        """
        Extract examiner information from the given text.

        Parameters:
        - text (str): Input text containing examiner information.

        Returns:
        - dict: Extracted examiner information or None if no match is found.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]

            # Extract fields based on the matched pattern
            title, first_name, last_name = None, None, None
            tokens = list(span)

            if pattern_name == "EXAMINER_INFO_1":
                title = tokens[3].text
                first_name = tokens[4].text
                last_name = tokens[5].text

            elif pattern_name == "EXAMINER_INFO_2":
                first_name = tokens[3].text
                last_name = tokens[4].text

            return {
                "title": title,
                "first_name": first_name,
                "last_name": last_name
            }

        return None

class EndoscopeDataExtractor:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_lg")
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        """Define patterns for extracting endoscope information."""
        # Pattern 1: Endoskop: Modellname Seriennummer: NNNNNNNN
        pattern1 = [
            {"LOWER": "endoskop"},
            {"TEXT": ":"},
            {"POS": "PROPN", "OP": "+"},
            {"LOWER": "seriennummer"},
            {"TEXT": ":"},
            {"SHAPE": "dddddddd"}
        ]

        self.matcher.add("ENDOSCOPE_INFO_1", [pattern1])

    def extract_endoscope_info(self, text):
        """
        Extract endoscope information from the given text.

        Parameters:
        - text (str): Input text containing endoscope information.

        Returns:
        - dict: Extracted endoscope information or None if no match is found.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]

            # Extract fields based on the matched pattern
            model_name, serial_number = None, None
            tokens = list(span)

            if pattern_name == "ENDOSCOPE_INFO_1":
                model_name = " ".join([t.text for t in tokens[2:tokens.index(tokens[3])]])
                serial_number = tokens[-1].text

            return {
                "model_name": model_name,
                "serial_number": serial_number
            }

        return None

class ExaminationDataExtractor:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_lg")
        
    def extract_examination_info(self, text, remove_examiner_titles=True):
        """
        Extracts examiner and examination time information from the given text.
        
        Parameters:
        - text (str): Text containing examination information.
        - remove_examiner_titles (bool): Whether to remove titles from examiner names.
        
        Returns:
        - dict: Extracted examination information or None if no match is found.
        """
        # Try to match the first pattern: "Unters.: [last name], [first name] U-datum: [date] [time]"
        if "1. Unters.:" in text or "Unters.:" in text:
            return self._extract_meta_format_1(text, remove_examiner_titles)
        
        # Try to match the second pattern: "Eingang am: [date]"
        if "Eingang am:" in text:
            return self._extract_meta_format_2(text, remove_examiner_titles)
            
        return None
        
    def _extract_meta_format_1(self, line, remove_examiner_titles=True):
        """Extract examiner info with date and time."""
        if remove_examiner_titles:
            # This would need implementation of remove_titles function
            pass
            
        # Define the regular expression pattern for matching the relevant fields
        pattern = r"Unters\.: ([\w\s\.]+), ([\w\s]+)\s*U-datum:\s*(\d{2}\.\d{2}\.\d{4}) (\d{2}:\d{2})"
        
        # Search for the pattern in the given line
        match = re.search(pattern, line)
        
        if match:
            examiner_last_name = match.group(1).strip()
            examiner_first_name = match.group(2).strip()
            
            # Convert the examination date to the format YYYY-MM-DD
            examination_date = datetime.strptime(match.group(3), '%d.%m.%Y').strftime('%Y-%m-%d')
            
            # Extract the examination time
            examination_time = match.group(4)
            
            return {
                'examiner_last_name': examiner_last_name,
                'examiner_first_name': examiner_first_name,
                'examination_date': examination_date,
                'examination_time': examination_time
            }
        else:
            data = ExaminerDataExtractor().extract_examiner_info(line)
            if data:
                return data
            else:
                return None
        
        return None
        
    def _extract_meta_format_2(self, line, remove_examiner_titles=True):
        """Extract only examination date."""
        if remove_examiner_titles:
            # This would need implementation of remove_titles function
            pass
            
        # Define the regular expression pattern for matching the relevant fields
        pattern = r"Eingang am:\s*(\d{2}\.\d{2}\.\d{4})"
        
        # Search for the pattern in the given line
        match = re.search(pattern, line)
        
        if match:
            # Convert the examination date to the format YYYY-MM-DD
            examination_date = datetime.strptime(match.group(1), '%d.%m.%Y').strftime('%Y-%m-%d')
            
            return {
                'examiner_last_name': "",
                'examiner_first_name': "",
                'examination_date': examination_date,
                'examination_time': ""
            }
        
        return None
