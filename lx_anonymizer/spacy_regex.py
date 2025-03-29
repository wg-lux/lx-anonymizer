import re
import spacy
from spacy.matcher import Matcher
from spacy.tests.pipeline.test_span_ruler import patterns
from datetime import datetime
import warnings

class PersonDataExtractor:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_sm")
        self.ruler = self.nlp.add_pipe("entity_ruler")
        self.setup_matcher()
    
    def setup_matcher(self):
        """Setup the SpaCy Matcher with patient information patterns"""
        self.matcher = Matcher(self.nlp.vocab)
        
        # Pattern 1: Patient: Nachname ,Vorname geb. DD.MM.YYYY Fallnummer: NNNNNNNN
        pattern1 = [
            {"LOWER": "patient"}, 
            {"LOWER": ":"}, 
            {"POS": "PROPN", "OP": "+", "NAME": "last_name"}, 
            {"TEXT": ","}, 
            {"POS": "PROPN", "OP": "+", "NAME": "first_name"}, 
            {"LOWER": "geb"}, 
            {"TEXT": "."}, 
            {"SHAPE": "dd.dd.dddd", "OP": "?", "NAME": "birthdate"},
            {"LOWER": "fallnummer"}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dddddddd", "NAME": "casenumber"}
        ]

        # Pattern 2: Patient: Nachname, Vorname geboren am: DD.MM.YYYY
        pattern2 = [
            {"LOWER": "patient"}, 
            {"LOWER": ":"}, 
            {"POS": "PROPN", "OP": "+", "NAME": "last_name"}, 
            {"TEXT": ","}, 
            {"POS": "PROPN", "OP": "+", "NAME": "first_name"}, 
            {"LOWER": "geboren"}, 
            {"LOWER": "am"}, 
            {"TEXT": ":"}, 
            {"SHAPE": "dd.dd.dddd", "OP": "?", "NAME": "birthdate"}
        ]
        
        # Add patterns to the matcher
        self.matcher.add("PATIENT_INFO_1", [pattern1])
        self.matcher.add("PATIENT_INFO_2", [pattern2])
    
    def patient_data_extractor(self, text):
        """Extract patient data using entity ruler"""
        doc = self.nlp(text)
        
        patterns = [
            r"Patient: (?P<last_name>[\w\s-]+) ,(?P<first_name>[\w\s-]+) geb\. (?:(?P<birthdate>\d{2}\.\d{2}\.\d{4}))? *Fallnummer: (?P<casenumber>\d+)",
            r"Patient: (?P<last_name>[\w\s-]+),\s?(?P<first_name>[\w\s-]+) geboren am: (?:(?P<birthdate>\d{2}\.\d{2}\.\d{4}))?"
        ]
        
        self.ruler.add_patterns(patterns)
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        return None
    
    def extract_patient_info_spacy(self, text):
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
            
            # Get pattern name to determine extraction logic
            pattern_name = self.nlp.vocab.strings[match_id]
            
            # Find the last name and first name tokens
            last_name_tokens = []
            first_name_tokens = []
            birthdate = None
            casenumber = None
            
            # Extract named entities from the matched span
            for i, token in enumerate(span):
                # Find last name (after "Patient:")
                if i > 1 and token.text == ",":  # Stop at comma
                    j = 2  # Start after "Patient:"
                    while j < i:
                        last_name_tokens.append(span[j].text)
                        j += 1
                
                # Find first name (after comma)
                if token.text == "," or token.text == " ,":
                    j = i + 1
                    while j < len(span) and not (span[j].lower_ == "geb" or span[j].lower_ == "geboren"):
                        first_name_tokens.append(span[j].text)
                        j += 1
                
                # Find birthdate
                if pattern_name == "PATIENT_INFO_1" and token.lower_ == "geb" and i+2 < len(span) and span[i+1].text == ".":
                    if i+2 < len(span) and span[i+2].shape_ == "dd.dd.dddd":
                        birthdate = span[i+2].text
                
                if pattern_name == "PATIENT_INFO_2" and token.lower_ == "am" and i+2 < len(span):
                    if i+2 < len(span) and span[i+2].shape_ == "dd.dd.dddd":
                        birthdate = span[i+2].text
                
                # Find case number
                if pattern_name == "PATIENT_INFO_1" and token.lower_ == "fallnummer" and i+2 < len(span):
                    casenumber = span[i+2].text
            
            # Clean up the extracted names
            last_name = " ".join(last_name_tokens).strip()
            first_name = " ".join(first_name_tokens).strip()
            
            # Convert birthdate format if available
            formatted_birthdate = None
            if birthdate:
                try:
                    formatted_birthdate = datetime.strptime(birthdate, '%d.%m.%Y').strftime('%Y-%m-%d')
                except ValueError:
                    formatted_birthdate = None
            
            # Return extracted information
            if last_name and first_name:
                return {
                    'patient_first_name': first_name,
                    'patient_last_name': last_name,
                    'patient_dob': formatted_birthdate,
                    'casenumber': casenumber
                }
        
        return None
        
    def examiner_data_extractor(self, text):
        nlp = spacy.load("de_core_news_sm")
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
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        return None

