import re
from datetime import datetime

def extract_name_from_patient_line(text):
    """Extracts first and last name from typical patient lines in various formats."""
    patterns = [
        # Format: Patient: Last name, First name geb. DD.MM.YYYY Fallnummer: XXXXXXXX
        r"(?:Patient:|Patientin:|Pat\.:|Pat:)[\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]*[,][\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]*(?:geb\.|geboren am:)[\s]*(\d{1,2}\.\d{1,2}\.\d{4})(?:.*?(?:Fallnummer:|Fallnr\.:)[\s]*(\d+))?",
        
        # Format: Patient: Last name First name geb. DD.MM.YYYY Fallnummer: XXXXXXXX
        r"(?:Patient:|Patientin:|Pat\.:|Pat:)[\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]+([a-zA-ZäöüÄÖÜß\-]+)[\s]*(?:geb\.|geboren am:)[\s]*(\d{1,2}\.\d{1,2}\.\d{4})(?:.*?(?:Fallnummer:|Fallnr\.:)[\s]*(\d+))?",
        
        # Format: Patient: Last name, First name
        r"(?:Patient:|Patientin:|Pat\.:|Pat:)[\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]*[,][\s]*([a-zA-ZäöüÄÖÜß\-]+)",
        
        # Format: Patient: Last name First name
        r"(?:Patient:|Patientin:|Pat\.:|Pat:)[\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]+([a-zA-ZäöüÄÖÜß\-]+)",

        # Format: Pat.: Last name, Dr. First name Geb.Dat.: DD.MM.YYYY
        r"(?:Pat\.:|Patient:)[\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]*,[\s]*(Dr\.)[\s]*([a-zA-ZäöüÄÖÜß\-]+)[\s]*Geb\.Dat\.:?[\s]*(\d{1,2}\.\d{1,2}\.\d{4})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            last_name = groups[0].strip() if groups[0] else "UNBEKANNT"
            title = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            first_name = groups[2].strip() if len(groups) > 2 and groups[2] else "UNBEKANNT"
            birthdate = None
            if len(groups) >= 4 and groups[3]:
                try:
                    birthdate = datetime.strptime(groups[3].strip(), '%d.%m.%Y').strftime('%Y-%m-%d')
                except (ValueError, AttributeError):
                    pass
            return first_name, last_name, birthdate, title
    
    # Standardwerte, wenn keine Übereinstimmung gefunden wird
    return "UNBEKANNT", "UNBEKANNT", None, None

def validate_patient_info(info_dict):
    """Überprüft und korrigiert Patientendaten.
    
    Args:
        info_dict (dict): Dictionary mit Patienteninformationen
        
    Returns:
        dict: Validiertes und korrigiertes Dictionary
    """
    if not isinstance(info_dict, dict):
        return {
            'first_name': "UNBEKANNT",
            'last_name': "UNBEKANNT",
            'birthdate': None,
            'casenumber': None,
            'gender': "Unbekannt"
        }
    
    # Stelle sicher, dass erforderliche Schlüssel existieren
    for key in ['first_name', 'last_name']:
        if key not in info_dict or not info_dict[key] or info_dict[key] == "NOT FOUND":
            info_dict[key] = "UNBEKANNT"
    
    # Normalisiere Geschlecht
    if 'gender' in info_dict:
        if info_dict['gender'] == "male":
            info_dict['gender'] = "Männlich"
        elif info_dict['gender'] == "female":
            info_dict['gender'] = "Weiblich"
        elif info_dict['gender'] in ["unknown", "NOT FOUND"]:
            info_dict['gender'] = "Unbekannt"
    else:
        info_dict['gender'] = "Unbekannt"
        
    return info_dict
