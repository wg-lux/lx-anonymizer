import re
from datetime import datetime

def extract_patient_info_from_text(text):
    # Dictionary to store extracted information
    info = {
        'patient_first_name': "Unknown",
        'patient_last_name': "Unknown",
        'patient_dob': None,
        'patient_gender_name': "Unknown",
        'casenumber': None
    }
    
    """
    A more robust patient information extraction using advanced regex patterns.
    This is a fallback method when the SpaCy extractors fail.
    
    Args:
        text (str): The full text from which to extract patient information
        
    Returns:
        dict: Dictionary with patient information or None if no match is found
    """
    
    # Try multiple patterns to find patient information
    
    # Pattern 1: Patient info with birth date and case number
    pattern1 = r"(?:Patient|Pat|Patientin|Pat\.):?\s*([A-Za-zäöüÄÖÜß\-]+)[,\s]+([A-Za-zäöüÄÖÜß\-]+)\s+(?:geb\.|geboren am|Geb\.Dat\.|geboren):?\s*(\d{1,2}\.\d{1,2}\.\d{4})(?:.*?(?:Fallnummer|Fallnr\.|Fall\.Nr\.|Fall-Nr):?\s*(\d+))?"
    match1 = re.search(pattern1, text, re.IGNORECASE)
    
    if match1:
        info['patient_last_name'] = match1.group(1).strip()
        info['patient_first_name'] = match1.group(2).strip()
        
        # Convert date format
        try:
            birth_date = datetime.strptime(match1.group(3), '%d.%m.%Y').strftime('%Y-%m-%d')
            info['patient_dob'] = birth_date
        except (ValueError, TypeError):
            pass
            
        # Extract case number if available
        if match1.group(4):
            info['casenumber'] = match1.group(4).strip()
            
        # Determine gender based on context
        # Check for gender indicators near the extracted name
        context_window = text[max(0, text.find(info['patient_first_name']) - 30):
                              min(len(text), text.find(info['patient_first_name']) + len(info['patient_first_name']) + 30)]
        
        if "Patientin" in context_window:
            info['patient_gender_name'] = "female"
        elif "Patient" in context_window and "Patientin" not in context_window:
            info['patient_gender_name'] = "male"
            
        return info
    
    # Pattern 2: Simple patient name only
    pattern2 = r"(?:Patient|Pat|Patientin|Pat\.):?\s*([A-Za-zäöüÄÖÜß\-]+)[,\s]+([A-Za-zäöüÄÖÜß\-]+)"
    match2 = re.search(pattern2, text, re.IGNORECASE)
    
    if match2:
        info['patient_last_name'] = match2.group(1).strip()
        info['patient_first_name'] = match2.group(2).strip()
        return info
    
    # Pattern 3: Full name in one group with title
    pattern3 = r"(?:Patient|Pat|Patientin|Pat\.):?\s*((?:Dr\.|Prof\.|Herr|Frau)?\s*[A-Za-zäöüÄÖÜß\-\s]+)"
    match3 = re.search(pattern3, text, re.IGNORECASE)
    
    if match3:
        full_name = match3.group(1).strip()
        parts = full_name.split()
        
        if len(parts) >= 2:
            # Assume first part is first name, rest is last name
            info['patient_first_name'] = parts[0]
            info['patient_last_name'] = " ".join(parts[1:])
            return info
    
    return info
