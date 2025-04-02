import re
from django.template.defaultfilters import first, last
import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
from datetime import datetime
import warnings
from .determine_gender import determine_gender

class PatientDataExtractorLg:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_lg")
        self.ruler = self.nlp.add_pipe("entity_ruler")
        self.setup_matcher()
    
    def setup_matcher(self):
        """Setup the SpaCy Matcher with patient information patterns"""
        self.matcher = Matcher(self.nlp.vocab)
        
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

        # Neue Muster für zusätzliche Formate
        pattern3 = [
            {"LOWER": "patient"}, 
            {"TEXT": ":"}, 
            {"POS": "PROPN", "OP": "+"}, 
            {"TEXT": ","}, 
            {"POS": "PROPN", "OP": "+"}, 
            {"TEXT": "geb"}, 
            {"TEXT": "."}, 
            {"SHAPE": "dd.dd.dddd"}, 
            {"TEXT": "fallnummer"}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dddddddd"}
        ]
        pattern4 = [
            {"LOWER": "patient"}, 
            {"TEXT": ":"}, 
            {"POS": "PROPN", "OP": "+"}, 
            {"TEXT": ","}, 
            {"POS": "PROPN", "OP": "+"}, 
            {"TEXT": "geb"}, 
            {"LOWER": "am"}, 
            {"SHAPE": "dd.dd.dddd"}, 
            {"TEXT": "pat.nr."}, 
            {"TEXT": "fall.nr."}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dddddddd"}
        ]

        # New pattern: Last name before first name, separated by a comma
        pattern5 = [
            {"LOWER": "patient"}, 
            {"TEXT": ":"}, 
            {"POS": "PROPN", "OP": "+"},  # Last name
            {"TEXT": ","}, 
            {"POS": "PROPN", "OP": "+"},  # First name
            {"LOWER": "geb"}, 
            {"TEXT": "."}, 
            {"SHAPE": "dd.dd.dddd", "OP": "?"},
            {"LOWER": "fallnummer"}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dddddddd"}
        ]

        # New pattern: Last name before first name, no comma
        pattern6 = [
            {"LOWER": "patient"}, 
            {"TEXT": ":"}, 
            {"POS": "PROPN", "OP": "+"},  # Last name
            {"POS": "PROPN", "OP": "+"},  # First name
            {"LOWER": "geb"}, 
            {"TEXT": "."}, 
            {"SHAPE": "dd.dd.dddd", "OP": "?"},
            {"LOWER": "fallnummer"}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dddddddd"}
        ]

        # New pattern: Last name before first name with title and birthdate
        pattern7 = [
            {"LOWER": "pat"}, 
            {"TEXT": "."}, 
            {"POS": "PROPN", "OP": "+"},  # Last name
            {"TEXT": ","}, 
            {"TEXT": "dr."},  # Title
            {"POS": "PROPN", "OP": "+"},  # First name
            {"LOWER": "geb.dat"}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dd.dd.dddd"}  # Birthdate
        ]

        # Additional patterns for more flexible matching
        pattern8 = [
            {"LOWER": {"IN": ["patient", "pat", "patientin", "pat."]}},
            {"IS_PUNCT": True, "OP": "?"},
            {"POS": "PROPN", "OP": "+"},  # Last name
            {"IS_PUNCT": True, "OP": "?"},
            {"POS": "PROPN", "OP": "+"}   # First name
        ]
        
        pattern9 = [
            {"TEXT": {"REGEX": "(?i)pat(ient)?\.?:?"}},
            {"POS": "PROPN", "OP": "+"}   # First part of name
        ]
        
        pattern10 = [
            {"ORTH": "Patient"},
            {"ORTH": ":"},
            {"TEXT": {"REGEX": "[A-Z][a-zäöüÄÖÜß\-]+"}},  # More flexible name matching
            {"ORTH": ",", "OP": "?"},
            {"TEXT": {"REGEX": "[A-Z][a-zäöüÄÖÜß\-]+"}},
        ]

        # Add patterns to the matcher
        self.matcher.add("PATIENT_INFO_1", [pattern1])
        self.matcher.add("PATIENT_INFO_2", [pattern2])
        self.matcher.add("PATIENT_INFO_3", [pattern3])
        self.matcher.add("PATIENT_INFO_4", [pattern4])
        self.matcher.add("PATIENT_INFO_5", [pattern5])
        self.matcher.add("PATIENT_INFO_6", [pattern6])
        self.matcher.add("PATIENT_INFO_7", [pattern7])
        self.matcher.add("PATIENT_INFO_8", [pattern8])
        self.matcher.add("PATIENT_INFO_9", [pattern9])
        self.matcher.add("PATIENT_INFO_10", [pattern10])
    
    def patient_data_extractor(self, text):
        """Extract patient data using entity ruler"""
        doc = self.nlp(text)
        ruler = EntityRuler(self.nlp)
        
        patterns = [
            r"Patient: (?P<last_name>[\w\s-]+) ,(?P<first_name>[\w\s-]+) geb\. (?:(?P<birthdate>\d{2}\.\d{2}\.\d{4}))? *Fallnummer: (?P<casenumber>\d+)",
            r"Patient: (?P<last_name>[\w\s-]+),\s?(?P<first_name>[\w\s-]+) geboren am: (?:(?P<birthdate>\d{2}\.\d{2}\.\d{4}))?"
        ]
        
        self.ruler.add_patterns(patterns)
        self.nlp.add_pipe(ruler, before="ner")
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        return None
        
    def extract_patient_info(self, text):
        """
        Extract patient information using SpaCy's matcher
        
        Parameters:
        - text: str
            Text containing patient information
            
        Returns:
        - dict or None
            Dictionary with patient information if found, None otherwise
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]

            # Extraktion basierend auf dem erkannten Muster
            last_name, first_name, birthdate, casenumber = None, None, None, None
            tokens = list(span)

            if pattern_name in ["PATIENT_INFO_1", "PATIENT_INFO_3", "PATIENT_INFO_5"]:
                last_name = " ".join([t.text for t in tokens[2:tokens.index(tokens[3])]])
                first_name = " ".join([t.text for t in tokens[tokens.index(tokens[3]) + 1:tokens.index(tokens[5])]])
                birthdate = tokens[-3].text if tokens[-3].shape_ == "dd.dd.dddd" else None
                casenumber = tokens[-1].text if tokens[-1].shape_ == "dddddddd" else None

            elif pattern_name in ["PATIENT_INFO_2", "PATIENT_INFO_4", "PATIENT_INFO_6"]:
                last_name = " ".join([t.text for t in tokens[2:tokens.index(tokens[3])]])
                first_name = " ".join([t.text for t in tokens[tokens.index(tokens[3]) + 1:tokens.index(tokens[5])]])
                birthdate = tokens[-3].text if tokens[-3].shape_ == "dd.dd.dddd" else None
                casenumber = tokens[-1].text if tokens[-1].shape_ == "dddddddd" else None

            elif pattern_name == "PATIENT_INFO_7":
                last_name = tokens[2].text
                first_name = tokens[5].text
                birthdate = tokens[-1].text

            # Formatierung des Geburtsdatums
            if birthdate:
                try:
                    birthdate = datetime.strptime(birthdate, "%d.%m.%Y").strftime("%Y-%m-%d")
                except ValueError:
                    birthdate = None

            # Rückgabe der extrahierten Informationen
            if first_name and last_name:
                return {
                    "patient_first_name": first_name,
                    "patient_last_name": last_name,
                    "patient_dob": birthdate,
                    "casenumber": casenumber,
                    "patient_gender": determine_gender(first_name)
                }

        # Standardwerte, wenn keine Daten gefunden wurden
        return {
            "patient_first_name": "NOT FOUND",
            "patient_last_name": "NOT FOUND",
            "patient_dob": "NOT FOUND",
            "casenumber": "NOT FOUND",
            "patient_gender": "NOT FOUND"
        }
    
    def _extract_using_entity_ruler(self, text):
        """Helper method to extract patient information using entity ruler patterns"""
        nlp = spacy.load("de_core_news_lg")
        patterns = [
            {"label": "PER", "pattern": "Patient: [A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patient: [A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patient: [A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patient: [A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patient: [A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z]." + "\s[A-z]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4}"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnummer: [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Pat.: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},   
            {"label": "PER", "pattern": "Patient: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"}, 
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4} Fallnr. [0-9]+"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ [A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4}"},
            {"label": "PER", "pattern": "Patientin: [A-Z][a-z]+,[A-Z][a-z]+ [A-Z][a-z]+ geb. [0-9]{2}\.[0-9]{2}\.[0-9]{4}"},
            ]
        ruler = EntityRuler(nlp)
        ruler.add_patterns(patterns)
        nlp.add_pipe(ruler, before="ner")
        doc = nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PER":
                # Try to extract first and last name from entity text
                parts = ent.text.split()
                if len(parts) >= 3:  # "Patient:" + first_name + last_name
                    first_name = parts[1]
                    last_name = parts[2]
                    gender = determine_gender(first_name)
                    return {
                        'patient_first_name': first_name,
                        'patient_last_name': last_name,
                        'birthdate': None,
                        'casenumber': None,
                        'patient_gender': gender
                    }
                elif len(parts) == 2:  # Nur 2 Teile gefunden
                    first_name = parts[1]
                    last_name = "UNKNOWN"
                    gender = determine_gender(first_name)
                    if gender == "male":
                        gender = "Männlich"
                    elif gender == "female":
                        gender = "Weiblich"
                    else:
                        gender = "Unbekannt"
                    
                    return {
                        'patient_first_name': first_name,
                        'patient_last_name': last_name,
                        'birthdate': None,
                        'casenumber': None,
                        'patient_gender': gender
                    }
        
        # Wenn nichts gefunden wurde, geben wir sinnvolle Standardwerte zurück
        return {
            'patient_first_name': "UNBEKANNT",
            'patient_last_name': "UNBEKANNT",
            'birthdate': None,
            'casenumber': None,
            'patient_gender': "Unbekannt"
        }
        
    def examiner_data_extractor(self, text):
        nlp = spacy.load("de_core_news_lg")
        doc = nlp(text)
        patterns = [
            {"label": "PER", "pattern": "Dr\.\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\med\.\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\med\.\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\med\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\med\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},
            {"label": "PER", "pattern": "Dr\.\med\.\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+"},  
        ]
        ruler = EntityRuler(nlp)
        nlp.add_pipe(ruler)
        ruler.add_patterns(patterns)
        
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        return None
    

