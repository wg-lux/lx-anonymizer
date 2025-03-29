from datetime import datetime
import re
from ..utils import determine_gender
import gender_guesser.detector as gender
import warnings
    
def extract_patient_info(line, gender_detector=None):
    """
    Extracts patient information from a given text line.

    Parameters:
    - line: str
        A line of text containing patient information.
    - gender_detector: Object, optional
        An object for determining gender based on the first name.

    Returns:
    - info: dict
        A dictionary containing the extracted information: first_name, last_name,
        birthdate (formatted as YYYY-MM-DD), casenumber, and gender.

    Example:
    Input line: "Patient: Dietrich ,Jimmy Joe geb. 06.01.1983 Fallnummer: 0015744097"
    Output: {'first_name': 'Jimmy Joe', 'last_name': 'Dietrich', 'birthdate': '1983-01-06', 'casenumber': '0015744097'}
    """
    # Define the regular expression pattern for matching the relevant fields
    # Using named groups for better readability
    patterns = [
        r"Patient: (?P<last_name>[\w\s-]+) ,(?P<first_name>[\w\s-]+) geb\. (?:(?P<birthdate>\d{2}\.\d{2}\.\d{4}))? *Fallnummer: (?P<casenumber>\d+)",
        r"Patient: (?P<last_name>[\w\s-]+),\s?(?P<first_name>[\w\s-]+) geboren am: (?:(?P<birthdate>\d{2}\.\d{2}\.\d{4}))?"
    ]  
    if not gender_detector:
        warnings.warn("Warning: No gender detector provided, using default detector.")
        # Initialize your gender detector here
        gender_detector = gender.Detector()

    # Search for the pattern in the given line
    for pattern in patterns:
        match = re.search(pattern, line)
    
        if match:
            # Extract named groups
            last_name = match.group('last_name').strip()
            first_name = match.group('first_name').strip()
            # Implement your own determine_gender function or use gender_detector
            patient_gender = determine_gender(first_name.split()[0], gender_detector)
            
            birthdate_str = match.group('birthdate')
            
            # Convert the birthdate to the format YYYY-MM-DD if available, otherwise use a default value
            birthdate = datetime.strptime(birthdate_str, '%d.%m.%Y').strftime('%Y-%m-%d') if birthdate_str else '1900-01-01'
            # case number is optional
            try:
                casenumber = match.group('casenumber')
            except: 
                casenumber = None
            
            info = {
                'patient_first_name': first_name,
                'patient_last_name': last_name,
                'patient_dob': birthdate,
                'casenumber': casenumber,
                'gender': patient_gender,
            }
            
            return info
    return None  # Return None if the pattern doesn't match